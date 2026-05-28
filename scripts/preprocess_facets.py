#!/usr/bin/env python3
"""
scripts/preprocess_facets.py
──────────────────────────────
CLI wrapper around src/pipeline/preprocessor.py

Usage:
    python scripts/preprocess_facets.py \
        --input data/facets.csv \
        --output data/facets_processed.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table
import pandas as pd
import re

app = typer.Typer()
console = Console()

DOMAIN_MAP = {
    'linguistic': ['language','sentence','spelling','grammar','vocabulary','brevity','storytelling','comprehension','auditory','listening skills','analogies','alphabetical','numeric filing','logical sequence','reading','verbal','memory for sounds','auditory memory','sequential memory','sentence structure','information retention','synthesis of information','frankness','concreteness','reliance on context'],
    'pragmatics': ['communication','collaboration','feedback','meeting deadlines','decision','leadership','negotiation','social interaction','non-verbal','eye-contact','cooperation','delegation','participation','encouraging participation','contribution to group','work styles','controlling reactions','managing emotions','judging consequences','evaluating solutions','input','activator','initiative','assertiveness','social boldness','talkativeness','outspokenness'],
    'safety': ['disrespect','dishonesty','hostility','drug','physical-violence','violence','passive-aggressive','irritabilit','aggression','impudence','brazenness','hatefulness','coarseness','harmfulness','rebelliousness','manipulat','cunningness','suspicion','psychoticism','sensationalism','ethnocentrism','self-righteousness','cantankerousness','disagreeableness','martyrdom','overprotectiveness','acidity','immaturity','slothfulness','inefficiency','impracticalness','hysteria','kink-interest'],
    'emotion': ['emotion','mood','happiness','sadness','anxiety','depression','empathy','anger','joy','fear','stress','burnout','sentiment','affect','compassion','warmth','boredom','frustrat','contentment','bliss','merriness','discontentment','emotionalism','joyfulness','peacefulness','vivacity','ardency','enthusiasm','high-spiritedness','moroseness','aloofness','hesitation','desperation','affection','warmheartedness','blissfulness','genialness','cordiality','drollness','quirkiness','general mood','negative affect','attachment','ego dissolution','comfort with vulnerability'],
    'personality': ['openness','conscientiousness','neuroticism','extraversion','agreeableness','self-esteem','self-efficacy','naivety','risktaking','impulsiv','perseverance','determination','submissiveness','social desirability','big five','hexaco','honesty-humility','perceiving','judging','collectedness','adaptability','flexibility','independence','individuality','conservatism','liberalism','conformity','conventional','mysteriousness','genuine','self-improvement','self perspective','self-directedness','selfcontrol','self-effacement','dauntlessness','bravery','courageousness','fearfulness','boldness','seeking approval','abasement','servility','submission','unassertiveness','patient care','exemplariness'],
    'cognitive': ['reasoning','intelligence','memory','attention','spatial','numerical','analytical','critical reasoning','synthesis','creativity','arithmetic','iq','working memory','cognitive','mathematical','data analysis','logical','estimating','material properties','network basics','computer skills','troubleshooting','rapid cognitive','divided attention','common-sense','economic reasoning','statistical','innovation','originality','ideas generated','creativity in solutions','creative resilience','creative risk-taking','abstract','analogies','preferred epistemology','comprehension of spoken'],
    'health_lifestyle': ['sleep','diet','nutrition','exercise','physical','pain','metabolic','caffeine','hormone','immune','fitness','dance','training-cycle','macronutrient','dietary','breakfast','snacking','cooking','culinary','eating habits','eco-tourism','sustainable-transport','commute','outdoors','museum','travel','passport','digital-nomad','choir','music-lessons','open-source','blog','skill-endorsements','peer-to-peer','gamified','cloud-backup','robotics','graffiti','home-security','processed-food','local-food','peer-collaboration','soft-skill training','volunteer','subscription count','vision-check','chronic pain','sleep apnea','basophil','parathyroid','polygenic','chromatin','serotonin','drug-use','inversion comfort'],
    'spiritual_values': ['spiritual','religious','meditation','prayer','sufi','buddhist','islamic','hindu','jewish','i ching','kabbalah','yoga','mindfulness','holiness','bahá','quran','sikh','kirtan','vrata','bhagavad','eightfold path','zohar','dhikr','seerah','new-age','channeling','aura','gnostic','archon','discernment','satya','pilgrimage','scripture','sacred text','connectedness','binding foundations','justice-mindedness','moral','ethical','dignity','decency','civility','multiculturalism','cultural identity','patriotism','value orientation','universalism','aesthetic appreciation','aestheticism','cultural intelligence'],
}

RUBRICS = {
    'linguistic': {1:'Communication is severely unclear or incoherent.',2:'Frequent issues noticeably impede understanding.',3:'Adequate clarity; generally understandable.',4:'Mostly clear and articulate with minor issues.',5:'Exceptionally clear, precise, and polished.'},
    'pragmatics': {1:'Response completely fails pragmatic expectations.',2:'Poorly aligned with conversational goals.',3:'Adequately meets pragmatic demands.',4:'Well-calibrated; cooperative and contextually apt.',5:'Exemplary pragmatic competence.'},
    'safety': {1:'Explicit harmful or unsafe content present.',2:'Subtle safety concern; potentially problematic.',3:'Borderline; minor concern.',4:'Largely safe with negligible caveats.',5:'Completely safe and responsible.'},
    'emotion': {1:'Emotional tone grossly inappropriate.',2:'Noticeable affective disconnect.',3:'Adequate emotional calibration.',4:'Good emotional intelligence.',5:'Outstanding emotional resonance.'},
    'personality': {1:'Trait expression severely dysfunctional.',2:'Trait underdeveloped or maladaptive.',3:'Moderate functional trait expression.',4:'Well-expressed adaptive trait.',5:'Exemplary trait expression.'},
    'cognitive': {1:'Reasoning deeply flawed or incoherent.',2:'Significant cognitive gaps or errors.',3:'Reasonable performance with limitations.',4:'Strong reasoning; mostly accurate.',5:'Exceptional cognitive quality.'},
    'health_lifestyle': {1:'Behavior clearly harmful or unhealthy.',2:'Below healthy norms with concerns.',3:'Meets baseline health standards.',4:'Above-average healthy behavior.',5:'Exemplary health-conscious behavior.'},
    'spiritual_values': {1:'Values dimension absent or deeply misaligned.',2:'Limited alignment with stated values.',3:'Adequate expression of core values.',4:'Clear and thoughtful values expression.',5:'Profound alignment; spiritually coherent.'},
}

POLARITY_LOW = ['disrespect','dishonesty','hostility','aggression','harm','passive-aggressive','impudence','brazen','hatred','coarsen','sensationalism','irritabilit','neuroticism','psychoticism','hysteria','burnout','depression','anxiety','fearfulness','immaturity','inefficiency','slothfulness','impractical','suspicious','manipulat','cantankerous','disagreeabl','rebelliousness','hatefulness','martyrdom','ethnocentrism','acidity','social desirability bias','compulsive','overprotective','seeking approval','abasement','servility','withdrawnness','aloofness','moroseness','inattentiveness','boredom susceptibility','processed-food','drug-use','physical-violence','chronic pain','sleep apnea','negative affect']
LESS_EVALUABLE = ['basophil','parathyroid','polygenic','chromatin','caffeine sensitivity gene','serotonin transporter','immune-response age','metabolic rate','macronutrient','fsh level','caffeine intake']

def infer_domain(name):
    nl = name.lower()
    best, best_score = 'personality', 0
    for domain, keywords in DOMAIN_MAP.items():
        score = sum(1 for kw in keywords if kw in nl)
        if score > best_score: best_score, best = score, domain
    return best

def infer_polarity(name):
    nl = name.lower()
    return 'lower_is_better' if any(kw in nl for kw in POLARITY_LOW) else 'higher_is_better'

def is_header(f):
    f = str(f).strip()
    if re.match(r'^\d+\.\s', f): return False
    return f.endswith(':')

def clean_name(f):
    return re.sub(r'^\d+\.\s*', '', str(f)).strip().rstrip(':').strip()

def make_id(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')[:60]

def requires_ctx(name):
    terms = ['coherence','discourse','relevance','social interaction','collaboration','reliance on context','listening']
    return any(t in name.lower() for t in terms)

def evaluable(name, domain):
    nl = name.lower()
    if any(k in nl for k in LESS_EVALUABLE): return False
    return domain in ['linguistic','pragmatics','safety','emotion','personality','cognitive','spiritual_values']


@app.command()
def preprocess(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
    chunk_size: int = typer.Option(20, "--chunk-size"),
):
    """Clean and enrich the raw Facets CSV."""
    df = pd.read_csv(input)
    facets_raw = df.iloc[:, 0].tolist()
    console.print(f"[yellow]Raw rows:[/yellow] {len(facets_raw)}")

    rows, seen = [], set()
    for f in facets_raw:
        f = str(f).strip()
        if not f or f == 'nan' or is_header(f): continue
        name = clean_name(f)
        if not name: continue
        domain = infer_domain(name)
        polarity = infer_polarity(name)
        rubric = RUBRICS[domain]
        fid = make_id(name)
        if fid in seen: continue
        seen.add(fid)
        desc = f"Assesses the degree to which {name.lower()} is expressed in the conversation turn."
        rows.append({
            'facet_id': fid, 'facet_name': name, 'domain': domain,
            'description': desc,
            'rubric_1': rubric[1], 'rubric_2': rubric[2], 'rubric_3': rubric[3],
            'rubric_4': rubric[4], 'rubric_5': rubric[5],
            'is_turn_level': True, 'polarity': polarity, 'weight': 1.0,
            'requires_context': requires_ctx(name),
            'evaluable_from_text': evaluable(name, domain),
            'score_direction': 'high=good' if polarity == 'higher_is_better' else 'low=good',
            'prompt_hint': f"Score [{name}] (domain: {domain}): {desc[:100]} Rate 1-5.",
            'chunk_group': len(rows) // chunk_size,
        })

    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)

    table = Table(title=f"Preprocessing Complete — {len(result)} facets")
    table.add_column("Domain"); table.add_column("Count", justify="right")
    for domain, count in result['domain'].value_counts().items():
        table.add_row(domain, str(count))
    console.print(table)
    console.print(f"[green]✓ Saved to {output}[/green]")


if __name__ == "__main__":
    app()
