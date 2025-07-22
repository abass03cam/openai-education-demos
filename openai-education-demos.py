#!/usr/bin/env python
# coding: utf-8

import os
from openai import OpenAI
from dotenv import load_dotenv
from agents import Agent, Runner

# Lade API-Key aus .env-Datei (nicht öffentlich committen!)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

use_model = "gpt-4o"

# Beispiel 1: Kindgerechte Zusammenfassung
def summary_for_kids():
    messages = [
        {"role": "system", "content": "Fasse die Inhalte kindgerecht für einen Zweitklässler zusammen."},
        {"role": "user", "content": (
            "Jupiter ist der fünfte Planet von der Sonne und der größte im Sonnensystem. "
            "Er besteht hauptsächlich aus Gas und hat viele Monde. Man kann ihn mit bloßem Auge am Himmel sehen."
        )}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=100
    )
    print("\n📘 Zusammenfassung für Kinder:\n", response.choices[0].message.content)

# Beispiel 2: Text in Emojis übersetzen
def translate_to_emojis():
    messages = [
        {"role": "system", "content": "Übersetze den folgenden Text ausschließlich in Emojis."},
        {"role": "user", "content": "Künstliche Intelligenz ist eine vielversprechende Technologie."}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        max_tokens=64
    )
    print("\n🤖 Emoji-Übersetzung:\n", response.choices[0].message.content)

# Beispiel 3: Unterrichtsplan mit GPT-4
def algebra_lesson_plan():
    messages = [
        {"role": "user", "content": "Erstelle einen Unterrichtsplan für das Distributivgesetz in der Algebra mit Beispielen und häufigen Fehlern."}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    print("\n📐 Algebra-Unterrichtsplan:\n", response.choices[0].message.content)

# Beispiel 4: Gedichtanalyse mit Agenten
async def poetry_analysis_with_agents():
    poet = Agent(
        name="Poet",
        instructions="Du bist ein Poet. Schreibe ein Gedicht im Stil einer Ballade über Obdachlose im Winter.",
        model=use_model
    )
    haiku_analyst = Agent(
        name="Haiku-Analyst",
        instructions="Du bist Helen, die Haiku-Flüsterin. Du analysierst Gedichte im Haiku-Stil.",
        model=use_model
    )
    free_verse_analyst = Agent(
        name="Free-Verse-Analyst",
        instructions="Du bist Fred, Experte für freie Verse. Du analysierst freie Gedichte freundlich und kreativ.",
        model=use_model
    )
    rhyme_analyst = Agent(
        name="Rhyme-Analyst",
        instructions="Du bist Ron, der Reim-Meister. Du gibst technisches Feedback zu Reimgedichten.",
        model=use_model
    )
    triage = Agent(
        name="Triage-Agent",
        instructions="Ordne das Gedicht einem passenden Analysten zu – Haiku, freier Vers oder Reimgedicht.",
        model=use_model,
        handoffs=[haiku_analyst, free_verse_analyst, rhyme_analyst]
    )
    result = await Runner.run(starting_agent=poet, input="Schreibe ein Gedicht über die Notlage Obdachloser im Winter.")
    print("\n📜 Verfasstes Gedicht:\n", result.final_output)

    poem = result.final_output
    result = await Runner.run(starting_agent=triage, input=f"Hier ist das Gedicht: {poem}")
    print("\n🧠 Analyse durch passenden Agenten:\n", result.final_output)

# Beispiel 5: Codeanalyse mit Agenten
async def code_review_with_agents():
    developer = Agent(
        name="Developer",
        instructions="Du bist ein Entwickler. Schreibe eine Python-Funktion zur Sortierung einer Liste.",
        model=use_model
    )
    performance = Agent(
        name="Performance-Analyst",
        instructions="Du bist Paul, Experte für effizienten Code. Prüfe den Code auf Performance.",
        model=use_model
    )
    security = Agent(
        name="Security-Analyst",
        instructions="Du bist Sara, Expertin für sichere Programmierung. Prüfe auf Sicherheitslücken.",
        model=use_model
    )
    style = Agent(
        name="Style-Analyst",
        instructions="Du bist Tom, Stil-Guru. Gib Feedback zu Lesbarkeit und Style.",
        model=use_model
    )
    triage = Agent(
        name="Triage-Agent",
        instructions="Analysiere den Code und leite ihn an den passenden Analysten weiter.",
        model=use_model,
        handoffs=[performance, security, style]
    )
    result = await Runner.run(starting_agent=developer, input="Schreibe eine Python-Funktion, die eine Liste von Zahlen sortiert.")
    print("\n🧑‍💻 Generierter Code:\n", result.final_output)

    code = result.final_output
    result = await Runner.run(starting_agent=triage, input=f"Hier ist der Code: {code}")
    print("\n🔍 Codeanalyse durch passenden Agenten:\n", result.final_output)

# Hauptfunktion synchron starten
def main():
    summary_for_kids()
    print("\n" + "="*50 + "\n")
    translate_to_emojis()
    print("\n" + "="*50 + "\n")
    algebra_lesson_plan()

if __name__ == "__main__":
    main()
