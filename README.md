# Claims Workflow Demo

A customer asks, “What paperwork is missing for my windshield claim?” The assistant remembers the claim while collecting verification details. Once it finds a matching account, it can explain the outstanding document and offer a summary. Python controls when the assistant can read a claim or write to the simulated outbox. A language model interprets free-form messages in the optional chat UI.

## Try the workflow in one minute

From this directory, with Python 3.12:

```sh
python3 -m insurance_claims.demo
python3 -m unittest discover -s tests -v
```

The demo uses scripted inputs and runs offline with the standard library. Watch when the assistant first discloses “itemized glass receipt” and when the outbox first contains a message. The tests cover partial and conflicting identity details, multiple matching claims, another customer's claim, document lookup, declined consent and repeat delivery.

## Try free-form chat

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s chat_tests -v
```

The chat test substitutes model interpretations and exercises consent recovery after a failed request. To launch the UI, set OPENAI_API_KEY and OPENAI_MODEL in your terminal. Choose a model available to your account that supports structured outputs with the Responses API. Live chat sends messages to the API and incurs usage charges. Use only the fictional details below.

```sh
python -m streamlit run insurance_claims/ui.py
```

Try these messages in order:

1. “Which documents do I need for CL-7101?”
2. “My name is Robin Vale. My email is robin@example.org and my phone is 202-555-0131.”
3. “What needs to appear on the receipt?”
4. “That's everything.” Then try “Skip the email.”

Start a new conversation and ask about CL-8201 using Robin's details. The claim belongs to the second fictional account, so the assistant should give no details. Ask about Robin's auto claim without an ID to see how it handles two possible matches.

## How the pieces fit

| Module | Responsibility |
| --- | --- |
| insurance_claims/llm.py | Converts a customer message into validated fields, hints, topics and consent decisions. |
| insurance_claims/conversation.py | Carries questions across turns and handles clarification, refusals and model errors. |
| insurance_claims/agent.py | Advances verification, claim selection, processing and summary consent. |
| insurance_claims/tools.py | Checks ownership, builds the selected claim context and writes the simulated summary. |
| insurance_claims/answering.py | Uses the selected record and document guidance to explain follow-up questions. |

All records are fictional. Matching three fields illustrates a workflow rule for the demo. A deployed service would need real authentication and access controls. Email and handoff are simulated, and session state is held in memory.
