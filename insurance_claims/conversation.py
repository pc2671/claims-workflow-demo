from dataclasses import replace

from .agent import (
    begin_post_process,
    handle_email_decision,
    handle_turn,
)
from .llm import (
    CallerRole,
    EmailDecision,
    ModelRequestError,
    Scope,
    Topic,
    Emotion,
    interpret_message,
)
from .state import Phase, SessionState
from .answering import answer_grounded_followup, FALLBACK_ANSWER
from .tools import (
    remember_case_hints, get_discussed_claims, redact_identity_text,
)

EMPATHY_MESSAGES = {
    Emotion.FRUSTRATED: (
        "I understand this process is frustrating. "
    ),
    Emotion.ANXIOUS: (
        "I hear that you're worried. We can take this one step at a time. "
    ),
    Emotion.ANGRY: (
        "I hear how upsetting this has been. "
    ),
    Emotion.CONFUSED: (
        "I can help clarify this. Let's take it one step at a time. "
    ),
}


SUMMARY_CHOICE = (
    "Would you like to send the email summary or skip it? "
    "In this demo, sending is simulated."
)
SUMMARY_EXPLANATION = (
    "The summary includes the claim answers we discussed, each discussed "
    "claim's recorded status, and follow-up items. It goes to the email "
    "address on your account. Verification fields and the full chat "
    "transcript are not copied into it. Known identity values and common "
    "SSN, phone, and email formats are redacted from the summary content. "
    "Sending is optional and requires your explicit agreement. "
)


def handle_post_process_question(state, turn, message):
    """Read-only clarification; any question requires a new consent turn."""
    state.email_consent = None
    claim_topics = [t for t in turn.topics if t != Topic.EMAIL_SUMMARY]
    parts = []
    if Topic.EMAIL_SUMMARY in turn.topics:
        parts.append(SUMMARY_EXPLANATION)
    if claim_topics or turn.case_hints:
        try:
            claims = get_discussed_claims(state)
        except (PermissionError, ValueError):
            return "I can't access that discussion safely. Please ask for human assistance. " + SUMMARY_CHOICE
        target_id = state.selected_case_id
        if turn.case_hints:
            candidates = claims
            for key, value in turn.case_hints.items():
                def matches(claim):
                    actual = (claim["created_at"][5:7] if key == "month" else
                              claim["created_at"][:4] if key == "year" else claim.get(key, ""))
                    expected = value.zfill(2) if key == "month" else value
                    return str(actual).casefold() == expected.casefold()
                candidates = [claim for claim in candidates if matches(claim)]
            if len(candidates) != 1:
                return (
                    "I couldn't identify a single previously discussed claim "
                    "on your account from that request. Please clarify which "
                    "discussed claim you mean, or finish this summary choice "
                    "before starting a new claim conversation. " + SUMMARY_CHOICE
                )
            target_id = candidates[0]["case_id"]
        # Keep the live selection and phase unchanged while answering about
        # a previously discussed, reauthorized claim.
        context_state = replace(state, selected_case_id=target_id)
        answer = answer_grounded_followup(context_state, message)
        answer = redact_identity_text(state, answer)
        parts.append(answer)
        if answer != FALLBACK_ANSWER:
            note = {"case_id": target_id, "topic": "grounded_followup", "answer": answer}
            if note not in state.discussion_notes:
                state.discussion_notes.append(note)
    return "\n\n".join(parts + [SUMMARY_CHOICE])


def chat(state: SessionState, message: str) -> str:
    empathy = ""

    def reply(text: str) -> str:
        response = empathy + text

        state.messages.append({
            "role": "user",
            "content": message,
        })
        state.messages.append({
            "role": "assistant",
            "content": response,
        })

        return response

    # A handoff stops automated claim processing.
    if state.human_requested:
        return reply(
            "Automated processing is paused for human assistance. "
            "This demo records the handoff request but does not "
            "connect to a live representative."
        )

    if state.post_process_complete:
        return reply(
            "The summary is already in the demo outbox."
            if state.email_sent else "Email was skipped. No message was sent."
        )

    previous_reply = next(
        (
            entry["content"]
            for entry in reversed(state.messages)
            if entry["role"] == "assistant"
        ),
        None,
    )

    try:
        turn = interpret_message(
            message,
            phase=state.phase.value,
            previous_assistant_message=previous_reply,
        )
    except ModelRequestError:
        # Do not advance or act on an unsuccessful interpretation.
        return (
            "I'm having trouble understanding the message right now. "
            "Please try again. No workflow action was taken."
        )
    empathy = EMPATHY_MESSAGES.get(turn.emotion, "")

    # Preserve early hints, but discard stale filters when switching.
    remember_case_hints(state, turn.case_hints)

    # Store new identity information without treating it as verified.
    state.sensitive_identity_values.update(state.identity_fields.values())
    state.sensitive_identity_values.update(turn.identity_fields.values())
    identity_changed = False

    for key, value in turn.identity_fields.items():
        if state.identity_fields.get(key) != value:
            state.identity_fields[key] = value
            identity_changed = True

    if turn.policy_number is not None:
        if state.policy_number != turn.policy_number:
            state.policy_number = turn.policy_number
            identity_changed = True

    if identity_changed:
        state.verified_fields.clear()
        state.verified_party_id = None
        state.selected_case_id = None
        state.phase = Phase.VERIFY_ID
        state.email_consent = None

    # Representatives need an authorization flow we haven't built yet.
    if (
        turn.requests_human
        or turn.caller_role == CallerRole.REPRESENTATIVE
    ):
        state.human_requested = True
        return reply(
            "I've recorded a request for human assistance. "
            "If you're calling for someone else, their authorization "
            "must also be checked before claim details are shared. "
            "This demo does not connect to a live representative."
        )

    # Remember claim questions asked before we can answer them.
    if (
        state.phase in {Phase.VERIFY_ID, Phase.RESOLVE_INTENT}
        and turn.scope in {Scope.IN_SCOPE, Scope.MIXED}
    ):
        for topic in turn.topics:
            if topic == Topic.EMAIL_SUMMARY:
                continue
            if topic.value not in state.pending_topics:
                state.pending_topics.append(topic.value)
        if any(
            topic in {Topic.DOCUMENTS, Topic.UNKNOWN}
            for topic in turn.topics
        ):
            if message not in state.pending_followup_questions:
                state.pending_followup_questions.append(message)

    prefix = ""

    if turn.scope in {Scope.OUT_OF_SCOPE, Scope.MIXED}:
        state.out_of_scope_attempts += 1
        prefix = (
            "I can help with insurance claims, but I can't answer "
            "unrelated questions. "
        )

        if state.out_of_scope_attempts >= 3:
            return reply(
                prefix
                + "Would you like help from a human representative "
                "with your insurance needs?"
            )

        if turn.scope == Scope.OUT_OF_SCOPE:
            return reply(prefix + "What insurance question can I help with?")

    if turn.refuses_verification and not state.is_verified:
        state.verification_refusals += 1

        if state.verification_refusals >= 2:
            state.human_requested = True

            return reply(
                prefix
                + "I won't keep asking you for identity details. "
                "Automated processing is paused for human assistance. "
                "Claim details remain protected until verification "
                "and any required authorization are completed. "
                "This demo records the handoff but does not connect "
                "to a live representative."
            )

        return reply(
            prefix
            + "Verification protects your private claim information. "
            "You don't have to use your SSN: you can choose three "
            "matching fields from full name, date of birth, phone, "
            "email, or SSN last four. "
            "You can also ask for a human representative."
        )

    if state.phase == Phase.POST_PROCESS:
        if not state.is_verified:
            state.email_consent = None
            state.verified_fields.clear()
            state.verified_party_id = None
            state.selected_case_id = None
            state.phase = Phase.VERIFY_ID
            return reply(prefix + "Your verification is no longer valid. "
                         "Please provide three matching identity fields "
                         "before we continue. No email was sent.")

        # A question takes precedence even if the same message says send.
        # Never answer a clarification and send in the same turn.
        if turn.topics or turn.case_hints:
            return reply(prefix + handle_post_process_question(state, turn, message))

        if turn.email_decision == EmailDecision.SEND:
            return reply(prefix + handle_email_decision(state, consent=True))
        if turn.email_decision == EmailDecision.SKIP:
            return reply(prefix + handle_email_decision(state, consent=False))
        return reply(prefix + SUMMARY_EXPLANATION + SUMMARY_CHOICE)

    # Only finish an already selected, verified case.
    if (
        turn.finished
        and state.phase == Phase.PROCESS_CASE
        and state.is_verified
        and not turn.case_hints
        and not turn.topics
    ):
        return reply(prefix + begin_post_process(state))
    # Answer follow-up questions about the current authorized claim.
    # Identity changes, handoffs, scope checks, and email decisions
    # have already been handled above this point.
    if (
        state.phase == Phase.PROCESS_CASE
        and state.is_verified
        and not turn.case_hints
        and not state.pending_topics
        and not state.pending_followup_questions
        and any(t != Topic.EMAIL_SUMMARY for t in turn.topics)
    ):
        answer = redact_identity_text(state, answer_grounded_followup(state, message))

        # Record actual answers for the email summary.
        # A model failure is not a substantive claim answer.
        if answer != FALLBACK_ANSWER:
            note = {
                "case_id": state.selected_case_id,
                "topic": "grounded_followup",
                "answer": answer,
            }

            if note not in state.discussion_notes:
                state.discussion_notes.append(note)

        return reply(prefix + answer)

    # Answer pending questions first, then any new questions.
    # Remove duplicates while preserving their order.
    topic_names = list(dict.fromkeys(
        state.pending_topics
        + [topic.value for topic in turn.topics if topic != Topic.EMAIL_SUMMARY]
    ))

    if not topic_names:
        topic_names = [Topic.OVERVIEW.value]

    answers = []

    for topic_name in topic_names:
        answer = handle_turn(
            state,
            case_hints=turn.case_hints or None,
            topic=topic_name,
        )
        answers.append(answer)

        # A verification or clarification prompt is not an answer
        # to the pending claim question. Keep it for a later turn.
        if state.phase != Phase.PROCESS_CASE:
            break

        if topic_name in state.pending_topics:
            state.pending_topics.remove(topic_name)

    # The controller must finish verification and select an owned claim
    # before any remembered detailed question reaches the answerer.
    if (
        state.phase == Phase.PROCESS_CASE
        and state.is_verified
        and state.pending_followup_questions
    ):
        remembered_question = "\n\n".join(
            state.pending_followup_questions
        )

        detailed_answer = answer_grounded_followup(
            state,
            remembered_question,
        )
        detailed_answer = redact_identity_text(state, detailed_answer)
        answers.append(detailed_answer)

        if detailed_answer != FALLBACK_ANSWER:
            note = {
                "case_id": state.selected_case_id,
                "topic": "grounded_followup",
                "answer": detailed_answer,
            }

            if note not in state.discussion_notes:
                state.discussion_notes.append(note)

            state.pending_followup_questions.clear()

    return reply(prefix + "\n\n".join(answers))