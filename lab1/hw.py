#!/usr/bin/env python3
"""
Word Association Game using NLTK WordNet similarity and semantic relations.
- Colorful, concise console UI powered by Rich
- Class-based, tidy structure
- Rewards points by semantic closeness (Wu–Palmer similarity) and relation match
- Hints powered by WordNet relations (synonyms, hypernyms, hyponyms, meronyms, holonyms, antonyms)

Run:
  python word_association_game.py
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

# NLTK imports
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    wn.ensure_loaded()


# ------------------------- Helpers ------------------------- #
lemmatizer = WordNetLemmatizer()
console = Console()


def _lemmatize_lower(w: str) -> str:
    w = (w or "").strip().lower()
    # Lemmatize as noun/verb/adjective in sequence; first successful change wins
    for pos in ("n", "v", "a"):
        lem = lemmatizer.lemmatize(w, pos)
        if lem != w:
            return lem
    return w


@dataclass
class RelationBundle:
    word: str
    synonyms: Set[str]
    antonyms: Set[str]
    hypernyms: Set[str]
    hyponyms: Set[str]
    meronyms: Set[str]  # part + substance + member
    holonyms: Set[str]  # part + substance + member
    definitions: List[str]

    def all_related(self) -> Set[str]:
        return (
            self.synonyms
            | self.antonyms
            | self.hypernyms
            | self.hyponyms
            | self.meronyms
            | self.holonyms
        )


class WordNetExplorer:
    """Collect WordNet relations for a word."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _lemmas_to_words(lemmas: Iterable[wn.lemma]) -> Set[str]:
        words: Set[str] = set()
        for lem in lemmas:
            name = lem.name().replace("_", " ").lower()
            words.add(name)
        return words

    def collect(self, word: str) -> RelationBundle:
        w = _lemmatize_lower(word)
        synsets = wn.synsets(w)
        synonyms: Set[str] = set()
        antonyms: Set[str] = set()
        hypernyms: Set[str] = set()
        hyponyms: Set[str] = set()
        meronyms: Set[str] = set()
        holonyms: Set[str] = set()
        definitions: List[str] = []

        for s in synsets:
            # synonyms
            synonyms |= self._lemmas_to_words(s.lemmas())
            # antonyms
            for lem in s.lemmas():
                antonyms |= self._lemmas_to_words(ant.name() for ant in lem.antonyms())
            # hypernyms / hyponyms
            for h in s.hypernyms():
                hypernyms |= self._lemmas_to_words(h.lemmas())
            for h in s.hyponyms():
                hyponyms |= self._lemmas_to_words(h.lemmas())
            # meronyms (part/substance/member) and holonyms
            for m in s.part_meronyms() + s.substance_meronyms() + s.member_meronyms():
                meronyms |= self._lemmas_to_words(m.lemmas())
            for h in s.part_holonyms() + s.substance_holonyms() + s.member_holonyms():
                holonyms |= self._lemmas_to_words(h.lemmas())
            definitions.append(s.definition())

        # Clean up: remove the original word if present only in synonyms to avoid trivial hints
        synonyms.discard(w)

        return RelationBundle(
            word=w,
            synonyms=synonyms,
            antonyms=antonyms,
            hypernyms=hypernyms,
            hyponyms=hyponyms,
            meronyms=meronyms,
            holonyms=holonyms,
            definitions=definitions,
        )


class SimilarityScorer:
    """Compute semantic similarity and assign points."""

    def __init__(self, base_method: str = "wup") -> None:
        self.base_method = base_method

    def _pairwise_similarity(self, w1: str, w2: str) -> float:
        s1 = wn.synsets(w1)
        s2 = wn.synsets(w2)
        if not s1 or not s2:
            return 0.0
        best = 0.0
        for a in s1:
            for b in s2:
                if self.base_method == "path":
                    sim = a.path_similarity(b) or 0.0
                else:  # default: Wu–Palmer
                    sim = a.wup_similarity(b) or 0.0
                if sim > best:
                    best = sim
        return float(best)

    def score(self, target: str, guess: str, relations: RelationBundle) -> Tuple[int, Dict[str, float]]:
        t = _lemmatize_lower(target)
        g = _lemmatize_lower(guess)
        sim = self._pairwise_similarity(t, g)
        # Base points: 0..100 rounded
        base_pts = round(sim * 100)

        # Relation bonuses
        bonus = 0
        rel_hit = ""
        if g in relations.synonyms:
            bonus += 20
            rel_hit = "synonym"
        elif g in relations.hypernyms:
            bonus += 15
            rel_hit = "hypernym"
        elif g in relations.hyponyms:
            bonus += 15
            rel_hit = "hyponym"
        elif g in relations.meronyms:
            bonus += 10
            rel_hit = "meronym"
        elif g in relations.holonyms:
            bonus += 10
            rel_hit = "holonym"
        elif g in relations.antonyms:
            bonus += 5
            rel_hit = "antonym"

        total = max(0, min(120, base_pts + bonus))
        breakdown = {
            "similarity": sim,
            "base_points": base_pts,
            "bonus": bonus,
            "total": total,
            "relation": rel_hit,
        }
        return total, breakdown

    @staticmethod
    def closeness_label(sim: float) -> str:
        if sim >= 0.9:
            return "[bold green]Perfect![/bold green]"
        if sim >= 0.75:
            return "[green]Very close[/green]"
        if sim >= 0.6:
            return "[yellow]Close[/yellow]"
        if sim >= 0.4:
            return "[yellow3]Related[/yellow3]"
        if sim > 0.2:
            return "[orange1]Weakly related[/orange1]"
        return "[red]Barely related[/red]"


class GameUI:
    def __init__(self, console: Console) -> None:
        self.console = console

    def banner(self) -> None:
        title = Text("Word Association Game", style="bold cyan")
        subtitle = Text("NLTK + WordNet Similarity", style="dim")
        self.console.print(Panel.fit(Text.assemble(title, "\n", subtitle)))

    def show_relations(self, rb: RelationBundle) -> None:
        table = Table(title=f"Hints for: [bold]{rb.word}[/bold]", expand=False)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Examples", style="white")

        def fmt(items: Set[str]) -> str:
            if not items:
                return "—"
            ex = sorted(items)
            return ", ".join(ex[:10]) + (" …" if len(ex) > 10 else "")

        table.add_row("Synonyms", fmt(rb.synonyms))
        table.add_row("Hypernyms", fmt(rb.hypernyms))
        table.add_row("Hyponyms", fmt(rb.hyponyms))
        table.add_row("Meronyms", fmt(rb.meronyms))
        table.add_row("Holonyms", fmt(rb.holonyms))
        table.add_row("Antonyms", fmt(rb.antonyms))
        self.console.print(table)

    def feedback(self, guess: str, breakdown: Dict[str, float]) -> None:
        sim = breakdown["similarity"]
        msg = SimilarityScorer.closeness_label(sim)
        rel = breakdown.get("relation")
        extra = f"  (relation: {rel})" if rel else ""
        self.console.print(f"{msg}  similarity={sim:.2f}  base={breakdown['base_points']}  bonus={breakdown['bonus']}  total=[bold]{breakdown['total']}[/bold]{extra}")


class GameEngine:
    STARTER_WORDS = [
        "car", "music", "tree", "dog", "computer", "city", "book", "river", "phone", "coffee"
    ]

    def __init__(self, rounds: int = 5) -> None:
        self.rounds = rounds
        self.ui = GameUI(console)
        self.explorer = WordNetExplorer()
        self.scorer = SimilarityScorer("wup")
        self.total_points = 0

    def pick_word(self) -> str:
        return random.choice(self.STARTER_WORDS)

    def run(self) -> None:
        self.ui.banner()
        console.print("Type [bold]:hint[/bold] for clues, [bold]:skip[/bold] to change the word, [bold]:quit[/bold] to exit.\n")

        round_idx = 1
        while round_idx <= self.rounds:
            target = self.pick_word()
            rel = self.explorer.collect(target)

            console.rule(f"Round {round_idx}/{self.rounds} — target: [bold magenta]{target}[/bold magenta]")

            while True:
                guess = Prompt.ask("Your related word")
                if not guess:
                    continue
                g = guess.strip().lower()
                if g in {":quit", ":q"}:
                    self._finish()
                    return
                if g == ":skip":
                    console.print("[dim]Skipping to a new word…[/dim]")
                    target = self.pick_word()
                    rel = self.explorer.collect(target)
                    console.rule(f"New target: [bold magenta]{target}[/bold magenta]")
                    continue
                if g == ":hint":
                    self.ui.show_relations(rel)
                    continue

                pts, breakdown = self.scorer.score(target, g, rel)
                self.total_points += pts
                self.ui.feedback(g, breakdown)
                break  # next round

            round_idx += 1

        self._finish()

    def _finish(self) -> None:
        console.rule()
        console.print(Panel.fit(f"Thanks for playing! Total points: [bold green]{self.total_points}[/bold green]", title="Game Over", border_style="green"))


if __name__ == "__main__":
    try:
        GameEngine(rounds=5).run()
    except KeyboardInterrupt:
        console.print("\n[dim]Bye![/dim]")
