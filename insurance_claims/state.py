from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"


@dataclass
class SessionState:
    # Current workflow phase
    phase: Phase = Phase.VERIFY_ID

    # Identity information supplied by the caller.
    # Collecting these values does NOT mean they are verified.
    identity_fields: dict[str, str] = field(default_factory=dict)
    policy_number: str | None = None

    # Set only by verification logic after matching stored records.
    verified_fields: set[str] = field(default_factory=set)
    verified_party_id: str | None = None

    # Information remembered even if volunteered during verification.
    intent: str | None = None
    case_hints: dict[str, str] = field(default_factory=dict)

    # Set after verification and case resolution.
    selected_case_id: str | None = None

    # None = not decided, True = send, False = skip.
    email_consent: bool | None = None
    email_sent: bool = False

    # Support recovery and escalation.
    verification_attempts: int = 0
    verification_refusals: int = 0
    out_of_scope_attempts: int = 0
    human_requested: bool = False

    # Retain supplied values for redaction even after an identity correction.
    sensitive_identity_values: set[str] = field(default_factory=set)

    # Conversation history for this session.
    messages: list[dict[str, str]] = field(default_factory=list)

    # Claim answers actually discussed during the conversation.
    discussion_notes: list[dict[str, str]] = field(default_factory=list)

    # Demo email storage; no real messages are delivered.
    outbox: list[dict[str, str]] = field(default_factory=list)

    post_process_complete: bool = False

    # Questions waiting for verification or claim selection.
    pending_topics: list[str] = field(default_factory=list)

    # Detailed questions waiting for verification or claim selection.
    pending_followup_questions: list[str] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        allowed_fields = {"name", "dob", "phone", "email", "ssn_last4"}
        matched_fields = self.verified_fields & allowed_fields

        return (
            self.verified_party_id is not None
            and len(matched_fields) >= 3
        )