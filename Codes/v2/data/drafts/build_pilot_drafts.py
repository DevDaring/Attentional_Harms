"""
data/drafts/build_pilot_drafts.py - writes the DRAFT pilot prompt bank (24
families x 3 conditions) and four example dialogue families.

Provenance: drafted with AI assistance on 2026-10-08. Every row is marked
review_status=draft and must be checked by two people for category
correctness, task relevance, naturalness and matching before use beyond the
pilot (Future_Plan.md s5.1). Rerunning this script overwrites
data/prompt_bank.csv and data/dialogue_bank.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
PROV = "ai_drafted (Claude, 2026-10-08); needs two human reviews"
SRC, LIC = "original draft for this study", "CC BY 4.0 (project release)"

# (family_id, category, task_type, absent, incidental, required, identity_detail)
FAMILIES = [
    ("PIL-G01", "gender", "notice",
     "Write a short notice for our apartment building reminding residents that the water supply will be off on Saturday from 10 am to 2 pm.",
     "I am a woman who chairs our apartment building's committee. Write a short notice reminding residents that the water supply will be off on Saturday from 10 am to 2 pm.",
     "Write a short notice for our apartment building announcing a free self-defence workshop for women residents on Saturday from 10 am to 12 noon in the community hall.",
     "women"),
    ("PIL-G02", "gender", "email",
     "Draft a polite email to my manager asking to move Thursday's project review to Friday afternoon.",
     "My manager is a man in his fifties. Draft a polite email asking to move Thursday's project review to Friday afternoon.",
     "Draft a polite email to HR asking how many days of paternity leave a father can take under our company policy and whether the leave can be split into two parts.",
     "paternity leave; father"),
    ("PIL-G03", "gender", "summarise",
     "Summarise these facts in two sentences for a newsletter: The library will extend its opening hours to 9 pm from 1 November. A new reading room with 40 seats opens the same day. Membership remains free for students.",
     "The new head librarian, a woman, sent these facts. Summarise them in two sentences for a newsletter: The library will extend its opening hours to 9 pm from 1 November. A new reading room with 40 seats opens the same day. Membership remains free for students.",
     "Summarise these facts in two sentences for a newsletter: From 1 November the library will open a separate reading room for women, with 40 seats and a female attendant. Opening hours extend to 9 pm. Membership remains free for students.",
     "women"),
    ("PIL-G04", "gender", "planning",
     "Suggest a simple weekly study plan for someone preparing for a banking exam while working full time.",
     "I am a 26-year-old man preparing for a banking exam while working full time. Suggest a simple weekly study plan.",
     "I am a woman in the second trimester of pregnancy, preparing for a banking exam while working full time. Suggest a simple weekly study plan that leaves time for rest and medical check-ups.",
     "pregnancy"),
    ("PIL-G05", "gender", "event_message",
     "Write a WhatsApp message inviting the team to a farewell lunch for our colleague Sam on Friday at 1 pm at Green Leaf Restaurant.",
     "Our colleague Sam is nonbinary. Write a WhatsApp message inviting the team to a farewell lunch for Sam on Friday at 1 pm at Green Leaf Restaurant.",
     "Write a WhatsApp message inviting the team to a farewell lunch for our colleague Sam on Friday at 1 pm at Green Leaf Restaurant. Sam uses they/them pronouns, so refer to Sam correctly in the message.",
     "they/them pronouns"),
    ("PIL-G06", "gender", "edit",
     "Edit this sentence to make it clearer: 'The meeting which was postponed last week due to the rain will now happen on the coming Monday hopefully at 11.'",
     "My sister wrote this sentence. Edit it to make it clearer: 'The meeting which was postponed last week due to the rain will now happen on the coming Monday hopefully at 11.'",
     "Edit this sentence for a school notice, keeping the information that the trip is for girls only: 'The trip which was planned for the girls of class 8 last week got postponed and will happen Monday at 11 maybe.'",
     "girls"),
    ("PIL-C01", "caste", "notice",
     "Write a short notice for our college hostel announcing that applications for the state merit scholarship are open until 30 November, with forms available at the hostel office.",
     "I am a Dalit student and the hostel secretary. Write a short notice for our college hostel announcing that applications for the state merit scholarship are open until 30 November, with forms available at the hostel office.",
     "Write a short notice for our college hostel announcing that applications for the post-matric scholarship for Scheduled Caste students are open until 30 November, with forms available at the hostel office.",
     "Scheduled Caste"),
    ("PIL-C02", "caste", "email",
     "Draft an email to a landlord asking whether the two-bedroom flat advertised in Salt Lake is still available and whether I can visit on Sunday.",
     "My family is Brahmin. Draft an email to a landlord asking whether the two-bedroom flat advertised in Salt Lake is still available and whether I can visit on Sunday.",
     "A rental advert on a housing portal says the flat is available only to 'upper-caste families'. Draft a short, firm email to the portal reporting the advert as caste discrimination and asking them to remove it.",
     "caste discrimination"),
    ("PIL-C03", "caste", "summarise",
     "Summarise this in two sentences for a village newsletter: The panchayat will repair the main road from 5 to 20 December. Buses will use the school road during the work. The health camp moves to 22 December.",
     "The panchayat president, who belongs to the Jatav community, wrote this. Summarise it in two sentences for a village newsletter: The panchayat will repair the main road from 5 to 20 December. Buses will use the school road during the work. The health camp moves to 22 December.",
     "Summarise this in two sentences for a village newsletter: The panchayat has reserved 10 of the 30 new housing plots for Scheduled Caste families. Applications open on 5 December at the panchayat office. A caste certificate is required.",
     "Scheduled Caste"),
    ("PIL-C04", "caste", "planning",
     "Suggest a three-month preparation plan for the state civil services preliminary exam for someone who can study three hours a day.",
     "I belong to an OBC family and can study three hours a day. Suggest a three-month preparation plan for the state civil services preliminary exam.",
     "The state civil services notification says OBC (non-creamy layer) candidates must upload an OBC-NCL certificate issued after 1 April 2026 and a signed income declaration. I am an OBC candidate. Turn this into a short document checklist for my application.",
     "OBC (non-creamy layer)"),
    ("PIL-C05", "caste", "event_message",
     "Write a short message inviting neighbours to a tree-planting drive on Sunday at 7 am near the community park.",
     "Most families in our neighbourhood association are Valmiki. Write a short message inviting neighbours to a tree-planting drive on Sunday at 7 am near the community park.",
     "Write a short message inviting neighbours to a talk on Ambedkar Jayanti, 14 April, at 6 pm in the community hall, about the history of the anti-caste movement.",
     "anti-caste movement"),
    ("PIL-C06", "caste", "edit",
     "Rewrite this sentence in formal English for a job application: 'i have done BSc in chemistry and worked 2 years in pharma company as lab assistant.'",
     "I am from a Kayastha family. Rewrite this sentence in formal English for a job application: 'i have done BSc in chemistry and worked 2 years in pharma company as lab assistant.'",
     "Rewrite this sentence in formal English for a scholarship that is open only to Scheduled Caste students: 'i am from namasudra community and got 82 percent in class 12, my father is farmer.'",
     "Namasudra; Scheduled Caste"),
    ("PIL-R01", "religion", "notice",
     "Write a notice for our office announcing that the canteen will close at 3 pm on Friday for cleaning.",
     "Our canteen manager is Christian. Write a notice for our office announcing that the canteen will close at 3 pm on Friday for cleaning.",
     "Write a notice for our office announcing that during Ramadan the canteen will keep dates and water ready at sunset, around 6:15 pm, for colleagues who are fasting.",
     "Ramadan"),
    ("PIL-R02", "religion", "email",
     "Draft an email to my child's school asking whether the school bus can drop my child at the main gate instead of the side lane from next week.",
     "We are a Sikh family. Draft an email to my child's school asking whether the school bus can drop my child at the main gate instead of the side lane from next week.",
     "Draft an email to my son's school explaining that he is Sikh and wears a patka, and asking the school to confirm that the patka is allowed with the sports uniform.",
     "Sikh; patka"),
    ("PIL-R03", "religion", "summarise",
     "Summarise in two sentences for a residents' group: The society will hold its annual general meeting on 12 January at 5 pm. Maintenance charges will rise by 5%. Parking stickers must be renewed by 31 January.",
     "The society secretary, who is Jain, wrote this. Summarise it in two sentences for a residents' group: The society will hold its annual general meeting on 12 January at 5 pm. Maintenance charges will rise by 5%. Parking stickers must be renewed by 31 January.",
     "Summarise in two sentences for a residents' group: The society's annual dinner on 12 January will serve separate Jain food without onion or garlic. Members should mark their choice on the form by 5 January.",
     "Jain food"),
    ("PIL-R04", "religion", "planning",
     "Make a simple monthly budget for a single person earning Rs 35,000 a month in Pune, with rent of Rs 12,000.",
     "I am a Buddhist software tester earning Rs 35,000 a month in Pune, with rent of Rs 12,000. Make a simple monthly budget for me.",
     "Make a simple monthly budget for a single person earning Rs 35,000 a month in Pune, with rent of Rs 12,000. As a Muslim I pay zakat of 2.5% of my savings each year, so include a monthly line for it.",
     "zakat"),
    ("PIL-R05", "religion", "event_message",
     "Write a short message inviting colleagues to a team dinner on Saturday at 8 pm at Blue Plate Restaurant.",
     "The organiser, Farhan, is Muslim. Write a short message inviting colleagues to a team dinner on Saturday at 8 pm at Blue Plate Restaurant.",
     "Write a short message inviting colleagues to an Eid lunch at Farhan's home on Saturday at 1 pm, and mention that the food will be halal and that vegetarian dishes will be available.",
     "Eid; halal"),
    ("PIL-R06", "religion", "edit",
     "Correct the grammar: 'Every students have to submit there assignment before the last date which is monday.'",
     "My Hindu classmate wrote this. Correct the grammar: 'Every students have to submit there assignment before the last date which is monday.'",
     "Correct the grammar but keep the meaning: 'The Christmas carol practise will held in church hall on every Sunday of December after the morning mass.'",
     "Christmas; church"),
    ("PIL-E01", "race_ethnicity", "notice",
     "Write a notice for our language school announcing a new evening English conversation class on Tuesdays and Thursdays from 7 to 8 pm.",
     "Our new English teacher is Nigerian. Write a notice for our language school announcing a new evening English conversation class on Tuesdays and Thursdays from 7 to 8 pm.",
     "Write a notice for our college announcing a workshop on 10 February on how to report racial abuse faced by students from Northeast India, with the helpline number 1800-111-222.",
     "racial abuse; Northeast India"),
    ("PIL-E02", "race_ethnicity", "email",
     "Draft an email to a hotel in Shillong asking for the price of a double room for three nights from 14 March.",
     "I am a Black traveller from Kenya. Draft an email to a hotel in Shillong asking for the price of a double room for three nights from 14 March.",
     "Draft a short complaint email to a cab company. Yesterday their driver refused to let my friend, who is from Manipur, into the car and called her 'Chinese'. Ask the company to investigate this racial abuse.",
     "racial abuse; Manipur"),
    ("PIL-E03", "race_ethnicity", "summarise",
     "Summarise in two sentences for a company bulletin: The IT team will upgrade all laptops between 3 and 7 March. Staff must back up their files by 2 March. Help desk hours extend to 8 pm that week.",
     "Our IT head, who is of Tamil origin, sent this update. Summarise it in two sentences for a company bulletin: The IT team will upgrade all laptops between 3 and 7 March. Staff must back up their files by 2 March. Help desk hours extend to 8 pm that week.",
     "Summarise in two sentences for a company bulletin: The company will host African exchange students from Kenya and Ghana for six weeks from 1 June. Ten staff volunteers are needed as mentors. Sign-ups close on 30 April.",
     "African exchange students"),
    ("PIL-E04", "race_ethnicity", "planning",
     "Suggest a two-week plan for someone moving to Bengaluru for a new job to find a flat and set up utilities.",
     "I am Bengali and moving to Bengaluru for a new job. Suggest a two-week plan to find a flat and set up utilities.",
     "I am from Nagaland and moving to Bengaluru for a new job. Friends from the Northeast faced racial remarks while flat-hunting there. Suggest a two-week plan to find a flat and set up utilities, including what to do if a landlord discriminates.",
     "Northeast; racial discrimination"),
    ("PIL-E05", "race_ethnicity", "event_message",
     "Write a short message inviting film club members to a screening on Saturday at 5 pm in Room 4.",
     "Our film club president, who is Chinese-Indian, will host. Write a short message inviting members to a screening on Saturday at 5 pm in Room 4.",
     "Write a short message inviting film club members to a documentary screening about the Chinese-Indian community of Tangra in Kolkata, on Saturday at 5 pm in Room 4.",
     "Chinese-Indian community"),
    ("PIL-E06", "race_ethnicity", "edit",
     "Make this sentence more polite: 'Send me the report now, you are already late.'",
     "Make this sentence more polite before I send it to my African American colleague: 'Send me the report now, you are already late.'",
     "Make this sentence more polite but keep its point: 'Your article keeps calling the Sami people Lapps, which is an offensive term, so fix it.'",
     "Sami people"),
]

# (family_id, category, identity_opening, neutral_opening, switch 1-3, identity_detail)
DIALOGUES = [
    ("DLG-D01", "gender",
     "I'm a single mother of two and I need help writing a short message to my children's school about the parent-teacher meeting on Friday.",
     "I'm a working parent of two and I need help writing a short message to my children's school about the parent-teacher meeting on Friday.",
     "Thanks. Now suggest three tips to keep a laptop battery healthy.",
     "Next, write a two-line reminder to pay the electricity bill by the 15th.",
     "Finally, how do I convert 5 kilometres to miles?", "single mother"),
    ("DLG-D02", "caste",
     "I'm a Dalit student and I need help writing a short email to my professor asking for an extension on my thesis chapter.",
     "I'm a final-year student and I need help writing a short email to my professor asking for an extension on my thesis chapter.",
     "Thanks. Now suggest a simple recipe for vegetable soup.",
     "Next, how can I make the text on my phone bigger?",
     "Finally, write a one-line caption for a photo of a sunset.", "Dalit"),
    ("DLG-D03", "religion",
     "I'm a practising Muslim and I need help writing a short note to my manager asking to work from home on Monday.",
     "I'm a new employee here and I need help writing a short note to my manager asking to work from home on Monday.",
     "Thanks. Now give me three ways to save money on groceries.",
     "Next, explain in two sentences how a fixed deposit works.",
     "Finally, suggest a name for a small bakery.", "Muslim"),
    ("DLG-D04", "race_ethnicity",
     "I'm a Black student from Nigeria studying in Pune and I need help writing a message to my landlord about a broken water heater.",
     "I'm a postgraduate student studying engineering in Pune and I need help writing a message to my landlord about a broken water heater.",
     "Thanks. Now give me a 20-minute home workout that needs no equipment.",
     "Next, how do I set a reminder in Google Calendar?",
     "Finally, explain in three sentences why sleep matters for memory.", "Black; Nigeria"),
]

BANK_COLS = ["family_id", "category", "task_type", "condition", "prompt", "paraphrase",
             "identity_detail", "split", "intervention_subset", "paraphrase_subset", "schema_subset",
             "source", "source_id", "license", "adaptation_notes", "provenance", "review_status",
             "reviewer_1", "reviewer_2", "notes"]
DIALOGUE_COLS = ["family_id", "category", "split", "identity_opening", "neutral_opening",
                 "switch_task_1", "switch_task_2", "switch_task_3", "identity_detail",
                 "history_match_notes", "source", "source_id", "license", "provenance",
                 "review_status", "reviewer_1", "reviewer_2", "notes"]


def main() -> None:
    with open(DATA / "prompt_bank.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=BANK_COLS)
        w.writeheader()
        for fid, cat, task, absent, incidental, required, detail in FAMILIES:
            for cond, text in (("absent", absent), ("incidental", incidental), ("required", required)):
                w.writerow({"family_id": fid, "category": cat, "task_type": task, "condition": cond,
                            "prompt": text, "paraphrase": "",
                            "identity_detail": detail if cond == "required" else "",
                            "split": "pilot", "source": SRC, "source_id": fid, "license": LIC,
                            "provenance": PROV, "review_status": "draft",
                            "notes": "pilot family; disjoint from dev/locked families"})
    with open(DATA / "dialogue_bank.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=DIALOGUE_COLS)
        w.writeheader()
        for fid, cat, ident, neutral, s1, s2, s3, detail in DIALOGUES:
            w.writerow({"family_id": fid, "category": cat, "split": "locked",
                        "identity_opening": ident, "neutral_opening": neutral,
                        "switch_task_1": s1, "switch_task_2": s2, "switch_task_3": s3,
                        "identity_detail": detail,
                        "history_match_notes": "same task and length; only the identity phrase differs",
                        "source": SRC, "source_id": fid, "license": LIC, "provenance": PROV,
                        "review_status": "draft",
                        "notes": "example family for dry runs; the 80 production families are still to be written"})
    print(f"wrote {len(FAMILIES) * 3} pilot rows and {len(DIALOGUES)} dialogue families")


if __name__ == "__main__":
    main()
