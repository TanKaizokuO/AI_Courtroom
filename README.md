In this AI-powered courtroom simulation, we have three GPT agents playing the roles of Prosecutor, Defense Attorney, and Judge. These agents are connected through a basic RAG (Retrieval-Augmented Generation) pipeline, which allows them to retrieve relevant legal principles to ground their arguments. The RAG pipeline works by having the agents generate a query based on the current state of the simulation, which is then used to retrieve relevant legal principles from a database. These principles are then used by the agents to generate more detailed and grounded arguments, which can be used to persuade the Judge in their favor. This simulation allows us to explore how AI agents can be used to simulate complex legal scenarios and how they can be used to generate arguments and legal principles in a more efficient and effective manner.

Circumstantial evidence can support a conviction if it is strong enough to point to the defendant as the perpetrator beyond a reasonable doubt. This means that the evidence must be consistent with the defendant's guilt and inconsistent with any other reasonable hypothesis. Additionally, the evidence must be sufficiently reliable and trustworthy to be considered credible.

Defendants are presumed innocent until proven guilty beyond a reasonable doubt. This means that the prosecution must prove the defendant's guilt beyond a reasonable doubt, and the defendant is not required to prove their innocence. The burden of proof lies solely on the prosecution.

Flight from the scene may be considered as consciousness of guilt if it is accompanied by other incriminating circumstances. For example, if the defendant flees the scene of a crime immediately after committing the crime, this may be seen as an attempt to evade detection or arrest, which could be considered consciousness of guilt. However, flight from the scene alone is not enough to prove guilt beyond a reasonable doubt.
## Final Verdict
**NOT GUILTY**

## Confidence Score
0.68
```

---

## ⚙️ Configuration

| Variable         | Default       | Description                 |
| ---------------- | ------------- | --------------------------- |
| `OPENAI_API_KEY` | _(required)_  | Your OpenAI API key         |
| `OPENAI_MODEL`   | `gpt-4o-mini` | Model to use for all agents |

Change `top_k` in `run_trial()` to retrieve more or fewer legal passages.

---

## 🔧 Extending

- **Add corpus entries**: Edit `LEGAL_CORPUS` in `rag.py`
- **Swap model**: Change `OPENAI_MODEL` in `.env`
- **Add agents**: Follow the pattern in `agents.py` — define a system prompt + a function
- **Swap retriever**: Replace the FAISS index with any vector DB (Chroma, Pinecone, etc.)






















Last updated: 2026-04-03 14:08:41
