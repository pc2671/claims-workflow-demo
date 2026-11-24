from .state import Phase, SessionState
from .tools import (
    PII_FIELDS,
    find_claims,
    get_selected_claim,
    verify_identity,
    send_summary_email,
    remember_case_hints,
    redact_identity_text,
)

FIELD_LABELS = {
    "name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "ssn_last4": "SSN last four digits",
}

ALLOWED_HINTS = {
    "case_id",
    "case_type",
    "status",
    "month",
    "year",
}

def _answer_claim_question(
    state: SessionState,
    topic: str,
) -> str:
    if state.phase != Phase.PROCESS_CASE:
        raise ValueError("Claim questions require the PROCESS_CASE phase.")

    claim = get_selected_claim(state)
    case_id = claim["case_id"]

    if topic == "overview":
        return (
            f"Claim {case_id}: {claim['summary']}. "
            "I can help with its status, denial reason, "
            "required documents, or payment information."
        )

    if topic == "status":
        return (
            f"Claim {case_id} currently has a status of "
            f"{claim['status']}."
        )

    if topic == "denial":
        reason = claim.get("denial_reason")

        if reason:
            return (
                f"The recorded denial reason for claim {case_id} "
                f"is that {reason}."
            )

        return (
            f"I don't see a denial reason recorded for claim {case_id}. "
            f"Its current status is {claim['status']}."
        )

    if topic == "documents":
        documents = claim.get("documents_needed")

        if documents:
            return (
                f"For claim {case_id}, the file lists these "
                f"required documents: {', '.join(documents)}."
            )

        return (
            f"I don't see required documents listed for claim {case_id}. "
            "That does not confirm that no additional documents "
            "will be requested."
        )

    if topic == "payment":
        amount = claim.get("net_pay")

        if amount is not None:
            return (
                f"The recorded amount paid for claim {case_id} "
                f"is ${amount} USD."
            )

        return (
            f"I don't see a payment amount recorded for claim {case_id}."
        )

    return (
        "I don't have enough information in the available claim data "
        "to answer that question. A human claims representative "
        "can help clarify it."
    )

def answer_claim_question(
    state: SessionState,
    topic: str,
) -> str:
    answer = redact_identity_text(state, _answer_claim_question(state, topic))

    note = {
        "case_id": state.selected_case_id,
        "topic": topic,
        "answer": answer,
    }

    if note not in state.discussion_notes:
        state.discussion_notes.append(note)

    return answer

def handle_turn(
    state: SessionState,
    *,
    identity_fields: dict[str, str] | None = None,
    policy_number: str | None = None,
    case_hints: dict[str, str] | None = None,
    topic: str = "overview",
) -> str:
    # These later-phase handlers will be implemented separately.
    if state.phase not in {
        Phase.VERIFY_ID,
        Phase.RESOLVE_INTENT,
        Phase.PROCESS_CASE,
    }:
        raise ValueError("This phase is not implemented yet.")

    # Remember clarification or start a fresh claim search.
    remember_case_hints(state, case_hints)

    # Collect identity fields. If an identity value changes after
    # verification, require verification again before accessing claims.
    state.sensitive_identity_values.update(state.identity_fields.values())
    state.sensitive_identity_values.update((identity_fields or {}).values())
    identity_changed = False

    for key, value in (identity_fields or {}).items():
        if key in PII_FIELDS and value.strip():
            value = value.strip()

            if state.identity_fields.get(key) != value:
                state.identity_fields[key] = value
                identity_changed = True

    if policy_number and policy_number.strip():
        value = policy_number.strip()

        if state.policy_number != value:
            state.policy_number = value
            identity_changed = True

    if identity_changed or not state.is_verified:
        state.phase = Phase.VERIFY_ID
        state.verified_fields.clear()
        state.verified_party_id = None
        state.selected_case_id = None
    
    if state.phase == Phase.PROCESS_CASE and not case_hints:
        return answer_claim_question(state, topic)


    if state.phase == Phase.VERIFY_ID:
        supplied = {
            key
            for key, value in state.identity_fields.items()
            if key in PII_FIELDS and value.strip()
        }

        if len(supplied) < 3:
            needed = 3 - len(supplied)
            options = [
                label
                for key, label in FIELD_LABELS.items()
                if key not in supplied
            ]

            return (
                "Verification protects your private claim information. "
                "Before I can access claim details, I need three "
                "matching identity fields. "
                f"Please provide {needed} more identity field(s) "
                f"from these remaining options: {', '.join(options)}. "
                "No particular field is mandatory—you can choose "
                "alternatives to your date of birth or SSN. "
                "You can also ask for a human representative."
            )

        if not verify_identity(state):
            return (
                "I couldn't verify your identity with those details. "
                "Please check what you provided and correct any errors. "
                "Claim details remain protected until verification succeeds."
            )

        state.phase = Phase.RESOLVE_INTENT

    # All claim lookup happens after verification.
    # Re-resolve when new hints arrive during PROCESS_CASE as well.
    state.phase = Phase.RESOLVE_INTENT
    state.selected_case_id = None

    if not state.case_hints:
        return (
            "Your identity is verified. "
            "Which claim can I help with? You can provide a claim ID, "
            "claim type, date, or status."
        )

    claims = find_claims(state)

    if not claims:
        return (
            "I couldn't find a claim on your account matching those details. "
            "Could you check the claim ID, type, date, or status?"
        )

    if len(claims) > 1:
        choices = "; ".join(
            f"{claim['case_id']} "
            f"({claim['case_type']}, {claim['created_at']}, "
            f"{claim['status']})"
            for claim in claims
        )

        return (
            f"I found more than one matching claim: {choices}. "
            "Which one would you like to discuss?"
        )

    claim = claims[0]
    state.selected_case_id = claim["case_id"]
    state.phase = Phase.PROCESS_CASE

    return answer_claim_question(state, topic)

def begin_post_process(state: SessionState) -> str:
    if state.phase != Phase.PROCESS_CASE:
        raise ValueError("Finish claim processing before offering email.")

    get_selected_claim(state)

    state.phase = Phase.POST_PROCESS
    state.email_consent = None
    state.post_process_complete = False

    return (
        "Would you like an email summary sent to the email address "
        "on your account? It will include what we discussed, "
        "each discussed claim’s recorded status, and follow-up items. "
        "You can choose to send it or skip it. "
        "In this demo, sending is simulated."
    )


def handle_email_decision(
    state: SessionState,
    consent: bool,
) -> str:
    if state.phase != Phase.POST_PROCESS:
        raise ValueError("Email must be offered before recording a decision.")

    # Reject ambiguous values such as strings or None.
    if type(consent) is not bool:
        raise ValueError("An explicit send or skip decision is required.")

    get_selected_claim(state)

    if state.post_process_complete:
        return (
            "The summary is already in the demo outbox."
            if state.email_sent
            else "Email was skipped. No message was sent."
        )

    state.email_consent = consent

    if not consent:
        state.post_process_complete = True
        return "No problem. We'll skip the email summary."

    try:
        send_summary_email(state)
    except (PermissionError, ValueError):
        state.email_consent = None
        return (
            "I couldn't safely prepare the summary. No email was sent. "
            "Please ask for human assistance or choose to skip the email."
        )

    return (
        "Your summary has been saved to the demo outbox. "
        "No real email was sent."
    )