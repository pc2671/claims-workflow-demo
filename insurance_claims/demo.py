"""Run a scripted conversation using only the Python standard library."""

from .agent import begin_post_process, handle_email_decision, handle_turn
from .state import SessionState


def main():
    state = SessionState()
    turns = [
        ("What paperwork is missing for CL-7101?", {"case_hints": {"case_id": "CL-7101"}}),
        ("My name is Robin Vale.", {"identity_fields": {"name": "Robin Vale"}}),
        ("My email is robin@example.org and my phone is 202-555-0131.",
         {"identity_fields": {"email": "robin@example.org", "phone": "202-555-0131"},
          "topic": "documents"}),
    ]
    for message, fields in turns:
        print(f"\nCustomer: {message}")
        print(f"Assistant: {handle_turn(state, **fields)}")
    print("\nAssistant:", begin_post_process(state))
    print("Customer: Yes, send the summary.")
    print("Assistant:", handle_email_decision(state, True))
    print("\nSimulated outbox\n", state.outbox[0]["body"], sep="")


if __name__ == "__main__":
    main()
