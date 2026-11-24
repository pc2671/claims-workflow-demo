import json
import re
from pathlib import Path

from .state import Phase, SessionState


FIXTURES_DIR = Path(__file__).parent / "fixtures"

PII_FIELDS = {"name", "dob", "phone", "email", "ssn_last4"}


def load_policyholders() -> list[dict]:
    path = FIXTURES_DIR / "policyholders.json"

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def normalize(field: str, value: str) -> str:
    value = value.strip()

    if field == "phone":
        digits = re.sub(r"\D", "", value)

        # Fixtures use US phone numbers with country code 1.
        if len(digits) == 10:
            digits = "1" + digits

        return digits

    if field in {"name", "email"}:
        return " ".join(value.casefold().split())

    return value


def matches_field(field: str, supplied: str, record: dict) -> bool:
    if field == "ssn_last4":
        # A national ID is not an SSN.
        if record.get("id_type") != "ssn_last4":
            return False

        stored_values = [record.get("id_last4", "")]
    else:
        stored_values = [
            record.get(field, ""),
            *record.get(f"{field}_aliases", []),
        ]

    supplied = normalize(field, supplied)

    return bool(supplied) and any(
        supplied == normalize(field, stored)
        for stored in stored_values
        if stored
    )


def verify_identity(state: SessionState) -> bool:
    # Clear previous verification results before evaluating.
    state.verified_fields.clear()
    state.verified_party_id = None

    supplied = {
        key: value
        for key, value in state.identity_fields.items()
        if key in PII_FIELDS and value.strip()
    }

    # Policy number does not count as one of the three PII fields.
    if len(supplied) < 3:
        return False

    state.verification_attempts += 1
    candidates = []

    for record in load_policyholders():
        if state.policy_number:
            if (
                state.policy_number.strip().upper()
                != record["policy_number"].upper()
            ):
                continue

        matched = {
            field
            for field, value in supplied.items()
            if matches_field(field, value, record)
        }

        # Require at least three matches, with no conflicting
        # supplied identity fields.
        if len(matched) >= 3 and matched == set(supplied):
            candidates.append((record, matched))

    # Ambiguous matches must not authorize access.
    if len(candidates) != 1:
        return False

    record, matched = candidates[0]
    state.verified_party_id = record["party_id"]
    state.verified_fields = matched

    return True

def load_claims() -> list[dict]:
    path = FIXTURES_DIR / "claims.json"

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def find_claims(state: SessionState) -> list[dict]:
    if not state.is_verified:
        raise PermissionError(
            "Identity verification is required before accessing claims."
        )

    # First restrict access to the verified policyholder.
    claims = [
        claim
        for claim in load_claims()
        if claim["party_id"] == state.verified_party_id
    ]

    # Then narrow the authorized claims using remembered hints.
    hints = state.case_hints

    for key in ("case_id", "case_type", "status"):
        value = hints.get(key)

        if value:
            claims = [
                claim
                for claim in claims
                if claim.get(key, "").casefold() == value.strip().casefold()
            ]

    month = hints.get("month")
    if month:
        claims = [
            claim
            for claim in claims
            if claim["created_at"][5:7] == month.strip().zfill(2)
        ]

    year = hints.get("year")
    if year:
        claims = [
            claim
            for claim in claims
            if claim["created_at"][:4] == year.strip()
        ]

    return claims

def get_selected_claim(state: SessionState) -> dict:
    # if it retrieves the selected claim and checks the authorization again
    if not state.is_verified:
        raise PermissionError(
            "Identity verification is required before accessing claims."
        )

    if not state.selected_case_id:
        raise ValueError("A claim must be selected first.")

    for claim in load_claims():
        if (
            claim["case_id"] == state.selected_case_id
            and claim["party_id"] == state.verified_party_id
        ):
            return claim

    raise ValueError("The selected claim is not available on this account.")

def redact_identity_text(state: SessionState, text: str) -> str:
    """Conservative demo redaction; not a general-purpose DLP system."""
    from datetime import date

    values = set(state.sensitive_identity_values)
    values.update(state.identity_fields.values())
    if state.policy_number:
        values.add(state.policy_number)
    # These stay local; do not send policyholder records to the answer model.
    for person in load_policyholders():
        for key in ("name", "dob", "phone", "email", "id_last4", "policy_number"):
            value = person.get(key)
            if value:
                values.add(value)
            values.update(person.get(f"{key}_aliases", []))

    patterns = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        patterns.append(re.escape(value).replace(r"\ ", r"\s+"))
        digits = re.sub(r"\D", "", value)
        if value.isdigit() and len(value) == 4:
            patterns.append(r"[\s.-]*".join(digits))
        if len(digits) in {10, 11} and re.fullmatch(r"[+()\d\s.-]+", value):
            phone = digits[-10:]
            patterns.append(r"(?:\+?1[\s().-]*)?" + r"[\s().-]*".join(phone))
        try:
            birthday = date.fromisoformat(value)
        except ValueError:
            continue
        patterns.extend([
            rf"0?{birthday.month}/0?{birthday.day}/{birthday.year}",
            rf"0?{birthday.month}-0?{birthday.day}-{birthday.year}",
            rf"{birthday.strftime('%B')}\s+0?{birthday.day},?\s+{birthday.year}",
            rf"{birthday.strftime('%b')}\.?\s+0?{birthday.day},?\s+{birthday.year}",
        ])

    for pattern in sorted(set(patterns), key=len, reverse=True):
        text = re.sub(r"(?<!\w)(?:" + pattern + r")(?!\w)", "[REDACTED]", text, flags=re.I)
    # Common formats also cover values never extracted into verification state.
    text = re.sub(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", "[REDACTED]", text)
    text = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[REDACTED]", text, flags=re.I)
    text = re.sub(r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\w)", "[REDACTED]", text)
    # Remove sentences that explicitly present an SSN or DOB value, including
    # spelled-out or unusual encodings. Over-redaction is preferable here.
    text = re.sub(
        r"[^.!?\n]*(?:\bSSN\b|social security|date of birth|\bDOB\b|birthday)[^.!?\n]*[.!?]?",
        "[Sensitive identity reference omitted]", text, flags=re.I | re.M,
    )
    return text


def get_discussed_claims(state: SessionState) -> list[dict]:
    """Reauthorize every referenced claim; never silently omit an invalid one."""
    selected = get_selected_claim(state)
    ids = list(dict.fromkeys(
        [note["case_id"] for note in state.discussion_notes]
        + [selected["case_id"]]
    ))
    owned = {
        claim["case_id"]: claim for claim in load_claims()
        if claim["party_id"] == state.verified_party_id
    }
    if any(case_id not in owned for case_id in ids):
        raise PermissionError("The discussion contains a claim unavailable on this account.")
    return [owned[case_id] for case_id in ids]


def send_summary_email(state: SessionState) -> dict[str, str]:
    """Build one redacted, multi-claim summary and simulate delivery."""
    if state.phase != Phase.POST_PROCESS:
        raise ValueError("Email requires the POST_PROCESS phase.")
    if state.email_consent is not True:
        raise PermissionError("Explicit email consent is required.")
    claims = get_discussed_claims(state)
    if state.email_sent:
        return state.outbox[-1]
    if state.post_process_complete:
        raise ValueError("Post-processing has already finished.")
    policyholder = next(
        (record for record in load_policyholders()
         if record["party_id"] == state.verified_party_id), None,
    )
    if not policyholder or not policyholder.get("email"):
        raise ValueError("No email address is available on the account.")

    sections = []
    for claim in claims:
        notes = list(dict.fromkeys(
            redact_identity_text(state, note["answer"])
            for note in state.discussion_notes
            if note["case_id"] == claim["case_id"]
        ))
        discussed = "\n".join(f"- {note}" for note in notes)
        if not discussed:
            discussed = "- No detailed claim answers were recorded."
        documents = claim.get("documents_needed", [])
        follow_up = (
            "The claim file lists these outstanding documents: "
            + ", ".join(documents)
            + ". Contact claims support to confirm the next steps."
            if documents else
            "No outstanding documents are listed in the available data. "
            "Contact claims support if further clarification is needed."
        )
        if claim.get("appeal_deadline"):
            follow_up += (
                f" Recorded appeal deadline: {claim['appeal_deadline']}. "
                "Confirm available options with claims support; "
                "no deadline extension is assumed."
            )
        sections.append(
            f"Claim: {claim['case_id']}\n"
            f"Recorded status: {claim['status']}\n\n"
            f"What we discussed:\n{discussed}\n\n"
            f"Follow-up items:\n{follow_up}"
        )
    subject = (
        f"Summary of claim {claims[0]['case_id']}" if len(claims) == 1
        else "Summary of claims " + ", ".join(c["case_id"] for c in claims)
    )
    body = "\n\n".join(sections) + (
        "\n\nThis conversation did not change the claim status "
        "or submit any documents."
    )
    message = {
        # Recipient is required for delivery; redaction applies to content.
        "to": policyholder["email"],
        "subject": redact_identity_text(state, subject),
        "body": redact_identity_text(state, body),
        "delivery": "simulated",
    }
    state.outbox.append(message)
    state.email_sent = True
    state.post_process_complete = True
    return message

def get_claim_answer_context(state: SessionState) -> dict:
    """Return authorized facts for grounded claim follow-up answers."""
    if state.phase not in {Phase.PROCESS_CASE, Phase.POST_PROCESS}:
        raise ValueError("Claim-answer context requires a processing phase.")
    if state.post_process_complete or state.human_requested:
        raise PermissionError("Automated claim answering has finished or paused.")

    # Recheck verification and ownership before loading answer context.
    claim = get_selected_claim(state)

    with (FIXTURES_DIR / "required_document_guideline.json").open(
        encoding="utf-8"
    ) as file:
        guidelines = json.load(file)

    with (FIXTURES_DIR / "claim_schema.json").open(
        encoding="utf-8"
    ) as file:
        schema = json.load(file)


    document_guidance = []

    for document in claim.get("documents_needed", []):
        lookup_key = document

        requirements = guidelines.get(
            "document_guidance", {}
        ).get(lookup_key, {}).get("en")

        alternatives = guidelines.get(
            "document_alternative_guidance", {}
        )
        alternative_text = alternatives.get(
            lookup_key,
            alternatives.get("default", {}),
        ).get("en")

        document_guidance.append({
            "requested_document": document,
            "requirements": requirements,
            "alternatives": alternative_text,
        })

    return {
        # Exclude the internal policyholder identifier.
        "claim": {
            key: value
            for key, value in claim.items()
            if key != "party_id"
        },
        "field_definitions": schema.get("field_descriptions", {}),
        "submission_guidance": guidelines.get(
            "default_guidance", {}
        ).get("en"),
        "case_type_guidance": guidelines.get(
            "case_type_guidance", {}
        ).get(claim["case_type"], {}).get("en"),
        "document_guidance": document_guidance,
        "followup_settings": guidelines.get(
            "claim_followup_settings", {}
        ),
        "grounding_rules": [
            "Use only the supplied claim facts and guidance.",
            "Do not promise approval, payment, or a status change.",
            "Do not invent submission links or contact details.",
            "General guidance does not override a recorded deadline.",
            "A guidance lookup alias does not change which document "
            "the claim actually requires.",
            "If the sources do not answer the question, say so and "
            "offer human assistance.",
        ],
    }

def remember_case_hints(
    state: SessionState,
    new_hints: dict[str, str] | None,
) -> None:
    allowed = {"case_id", "case_type", "status", "month", "year"}

    hints = {
        key: value.strip()
        for key, value in (new_hints or {}).items()
        if key in allowed and isinstance(value, str) and value.strip()
    }

    if not hints or state.phase == Phase.POST_PROCESS:
        return

    explicit_id = hints.get("case_id")
    processing_selected_claim = (
        state.phase == Phase.PROCESS_CASE
        and state.selected_case_id is not None
    )

    # Without an explicit ID, changed criteria during processing
    # trigger a fresh search rather than retaining stale filters.
    changed_criteria = any(
        value.casefold() != state.case_hints.get(key, "").casefold()
        for key, value in hints.items()
    )

    switching_claim = processing_selected_claim and (
        (
            explicit_id is not None
            and explicit_id.casefold()
            != state.selected_case_id.casefold()
        )
        or (
            explicit_id is None
            and changed_criteria
        )
    )

    if explicit_id is not None or switching_claim:
        state.case_hints.clear()

    state.case_hints.update(hints)

    if switching_claim:
        state.selected_case_id = None
        state.pending_topics.clear()
        state.pending_followup_questions.clear()
        state.email_consent = None

        state.phase = (
            Phase.RESOLVE_INTENT
            if state.is_verified
            else Phase.VERIFY_ID
        )