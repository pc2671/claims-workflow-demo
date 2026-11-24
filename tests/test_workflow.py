import unittest

from insurance_claims.agent import begin_post_process, handle_email_decision, handle_turn
from insurance_claims.state import Phase, SessionState
from insurance_claims.tools import get_claim_answer_context, send_summary_email


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.state = SessionState()
        self.identity = {"name": "Robin Vale", "email": "robin@example.org", "phone": "2025550131"}

    def select(self, case="CL-7101"):
        return handle_turn(self.state, identity_fields=self.identity,
                           case_hints={"case_id": case}, topic="documents")

    def test_partial_identity_defers_lookup_and_remembers_case(self):
        reply = handle_turn(self.state, identity_fields={"name": "Robin Vale"},
                            case_hints={"case_id": "CL-7101"})
        self.assertNotIn("receipt", reply)
        self.assertIsNone(self.state.selected_case_id)
        self.assertIn("receipt", handle_turn(self.state, identity_fields=self.identity, topic="documents"))

    def test_conflicting_fourth_field_blocks_verification(self):
        self.identity["dob"] = "2000-01-01"
        self.select()
        self.assertFalse(self.state.is_verified)
        self.assertIsNone(self.state.selected_case_id)

    def test_other_account_claim_is_hidden(self):
        self.assertNotIn("wheel", self.select("CL-8201"))
        self.assertIsNone(self.state.selected_case_id)
        self.state.selected_case_id = "CL-8201"
        self.state.phase = Phase.PROCESS_CASE
        with self.assertRaises(ValueError):
            get_claim_answer_context(self.state)

    def test_ambiguous_selection(self):
        reply = handle_turn(self.state, identity_fields=self.identity, case_hints={"case_type": "auto"})
        self.assertIn("CL-7101", reply)
        self.assertIn("CL-7102", reply)
        self.assertNotIn("CL-8201", reply)
        self.assertIsNone(self.state.selected_case_id)

    def test_document_guidance_uses_selected_claim(self):
        self.select()
        context = get_claim_answer_context(self.state)
        self.assertIn("shop name", context["document_guidance"][0]["requirements"])
        self.assertNotIn("party_id", context["claim"])

    def test_skip_leaves_empty_outbox(self):
        self.select()
        begin_post_process(self.state)
        with self.assertRaises(PermissionError):
            send_summary_email(self.state)
        handle_email_decision(self.state, False)
        self.assertEqual(self.state.outbox, [])

    def test_consent_creates_one_summary(self):
        self.select()
        begin_post_process(self.state)
        handle_email_decision(self.state, True)
        email = send_summary_email(self.state)
        self.assertEqual(len(self.state.outbox), 1)
        self.assertEqual(email["delivery"], "simulated")
        self.assertIn("receipt", email["body"])
        self.assertNotIn("robin@example.org", email["body"])
