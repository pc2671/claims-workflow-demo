import unittest
from unittest.mock import patch

from insurance_claims.conversation import chat
from insurance_claims.llm import EmailDecision, InterpretedTurn, ModelRequestError, Topic
from insurance_claims.state import Phase, SessionState


class ChatTests(unittest.TestCase):
    @patch("insurance_claims.conversation.interpret_message")
    def test_verified_conversation_with_consent_retry(self, interpret):
        state = SessionState()
        interpret.side_effect = [
            InterpretedTurn(case_hints={"case_id": "CL-7101"}, topics=[Topic.DOCUMENTS]),
            InterpretedTurn(identity_fields={"name": "Robin Vale", "phone": "2025550131",
                                            "email": "robin@example.org"}),
            InterpretedTurn(finished=True),
            ModelRequestError("Timeout"),
            InterpretedTurn(email_decision=EmailDecision.SEND),
        ]
        self.assertNotIn("glass receipt", chat(state, "What do I need for CL-7101?"))
        self.assertIn("glass receipt", chat(state, "Robin Vale, 2025550131, robin@example.org"))
        chat(state, "That's everything.")
        self.assertEqual(state.phase, Phase.POST_PROCESS)
        self.assertIn("No workflow action", chat(state, "Yes, send it."))
        self.assertEqual(state.outbox, [])
        chat(state, "Please send the summary.")
        chat(state, "Send it again.")
        self.assertEqual(len(state.outbox), 1)
