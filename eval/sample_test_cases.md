# Sample retrieval test cases (Legixo sample corpus)

**Purpose:** Optional **gold-style** questions and answers for the **fictional** markdown corpus (`01_` … `06_` files). Use this to **check retrieval and grounding** against a known target.

**Important**

- All facts are **made up** for the exercise.
- **Exact chunk IDs** depend on how text is chunked; we list **source files** that should be cited.
- Answers may be **phrased differently**; judge whether the **same facts** appear and citations point to the **right documents**.

---

## In-corpus questions (answer from documents only)

| # | Question | Sample answer (facts to hit) | Expected source file(s) |
|---|----------|------------------------------|-------------------------|
| 1 | What notice period applies when Bluecrest or Priya Nambiar ends the employment agreement? | **60 days** written notice; during notice the employee must return laptops, badges, and source code access. | `02_employment_agreement_excerpt.md` |
| 2 | How long is the non-compete after leaving Bluecrest, and when does it apply? | **12 months** after leaving; applies when working for a **direct competitor** in the **same city** as a Bluecrest office, if the role uses the **same client list** supplied by Bluecrest. | `02_employment_agreement_excerpt.md` |
| 3 | What kinds of information are called out as confidential in the Bluecrest excerpt? | **Pricing sheets**, **unreleased product roadmaps**, and **customer names** marked "confidential" in writing. | `02_employment_agreement_excerpt.md` |
| 4 | What is the civil suit number and who are the parties in the transport invoice dispute memo? | Suit **CV-2024-8812**; **Arvind Mehta** (plaintiff) v. **Northfield Logistics Pvt. Ltd.** (defendant); dispute over unpaid invoices March–June 2024; defendant raises **damaged goods / offsets**. | `01_matter_memo_arvind_v_northfield.md` |
| 5 | Under the memo, what limitation period applies to contract claims under the fictional Riverside Code? | **Three years** from the **breach date**. | `01_matter_memo_arvind_v_northfield.md` |
| 6 | When is the next hearing in Arvind Mehta v. Northfield, and what is scheduled? | **15 August 2025**; **plaintiff's witness (billing head)** to be examined. | `01_matter_memo_arvind_v_northfield.md` |
| 7 | How many clear days before the listed date must parties file written arguments under the hearing notice rules? | **Seven clear days**; late filings may not be read without leave of court. | `03_hearing_notice_template.md` |
| 8 | What time is case CV-2024-8812 listed, and what is it for? | **11:00 a.m.**; arguments on **invoice set-off**. | `03_hearing_notice_template.md` |
| 9 | What happened to case CV-2023-4401 (Lakeview Society v. City Water Board), and what is the next date? | **Adjourned**; next date **22 September 2025** (interim relief on water supply). | `03_hearing_notice_template.md` |
|10 | For commercial suits above five lakh fictional rupees, what does Section 14 say about mediation? | **Mandatory mediation** for **30 days** unless **both parties waive in writing**. | `04_statute_style_excerpt_fictional.md` |
|11 | If a contract fixes no interest rate, what rate may be awarded on admitted dues under Section 22? | **Simple interest at 9% per year** from the **date of demand** until **payment**. | `04_statute_style_excerpt_fictional.md` |
|12 | What settlement offer did Northfield make in the counsel notes, and what counter-instruction did the client give? | Northfield offered **70% of open invoices** if Arvind drops the **damage counterclaim**; client to **counter at 85%** and keep the witness for the next hearing if **no agreement by 1 August 2025**. | `05_counsel_notes_settlement.md` |
|13 | Are the settlement talks described in the counsel notes binding? What is the reminder? | Talks are **without prejudice** unless a **signed term sheet** exists. | `05_counsel_notes_settlement.md` |
|14 | Who is the lessor and lessee for Unit 4B at Harbor View Tower, and what is the monthly rent? | **Lessor:** Kiran Patel; **Lessee:** Harbor Bean Roasters OPC; monthly rent **₹45,000**. | `06_property_lease_clause.md` |
|15 | What is the security deposit amount, and within how many days must it be refunded after handover? | Deposit **₹1,35,000** (three months' rent); refundable within **45 days** of handover if no damage beyond normal wear. | `06_property_lease_clause.md` |
|16 | Is subletting allowed for the Harbor View lease without extra steps? | **Not allowed** without **written consent** of the lessor. | `06_property_lease_clause.md` |

---

## Out-of-corpus questions (should not invent an answer from these docs)

| # | Question | Expected behavior | Why |
|---|----------|-------------------|-----|
| O1 | What is the population of Riverside city? | Say **not found in corpus** / **cannot answer from documents** (or similar). | Not in any sample file. |
| O2 | What penalty applies if Priya breaches the non-compete? | **Not stated** in the corpus; must **not** invent penalties. | Employment file has non-compete terms only, no penalty section. |
| O3 | Who won case CV-2024-8812? | **Not stated**; dispute is ongoing. | Memo describes stage and hearing, not outcome. |

---

## How to use this

1. After ingest, call `POST /ask` with each **in-corpus** question.
2. Check: answer includes the **key facts** in the sample column (wording can differ).
3. Check: **citations** include at least one of the listed **source files** (paths may be relative).
4. For **O1–O3**, check: **no fabricated citations** and a clear **refusal or not-found** message.

The machine-readable copy is in `sample_test_cases.json`.
