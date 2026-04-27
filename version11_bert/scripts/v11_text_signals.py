"""Lightweight text signals used only for explanations and dashboard context.

These are not used as BERT training features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalRule:
    name: str
    reason_tr: str
    pattern: str


SIGNAL_RULES: tuple[SignalRule, ...] = (
    SignalRule(
        "crypto_promo",
        "Kripto/pump/giveaway veya token promosyon sinyali var",
        r"(?:\$[a-z0-9]{2,12}\b|\bcrypto\b|\btoken\b|\bairdrop\b|\bbtc\b|\beth\b|\bwallet\b|\bmoon\b|\bpump\b|\bsignal\b|\bclaim\b|\bgiveaway\b)",
    ),
    SignalRule(
        "telegram_dm",
        "Telegram/DM/kanal daveti sinyali var",
        r"\b(?:telegram|t\.me|dm me|join(?: our)? channel|discord\.gg|whatsapp)\b",
    ),
    SignalRule(
        "gambling",
        "Bahis/kumar/kupon sinyali var",
        r"\b(?:bet|casino|gambling|odds|freebet|sportsbook|vip picks|prizepicks|fanduel|draftkings)\b",
    ),
    SignalRule(
        "engagement_bait",
        "Takip/RT/beğeni/çekiliş odaklı etkileşim sinyali var",
        r"\b(?:follow|retweet|repost|like and share|tag friends|win a|giveaway|sweepstakes|claim now)\b",
    ),
    SignalRule(
        "political_fear",
        "Siyasi korku, kampanya veya sert mobilizasyon dili sinyali var",
        r"\b(?:traitor|treason|deep state|fake news|witch hunt|ww3|invasion|illegals|enemy of the people|dictator|coup)\b",
    ),
    SignalRule(
        "dehumanizing",
        "Ayrımcı/dehumanize edici saldırı dili sinyali var",
        r"\b(?:vermin|subhuman|animals|parasites|predator|terrorist sympathizer)\b",
    ),
    SignalRule(
        "commerce_spam",
        "Kupon/indirim/satış çağrısı sinyali var",
        r"\b(?:coupon|discount|promo code|black friday|limited time|buy now|shop now|extra off)\b",
    ),
)


def compact_text(value: str | None) -> str:
    return " ".join(str(value or "").split())


def extract_signals(text: str) -> dict[str, object]:
    normalized = compact_text(text)
    lower = normalized.lower()
    signals: list[str] = []
    reasons: list[str] = []
    matched_terms: list[str] = []
    for rule in SIGNAL_RULES:
        matches = re.findall(rule.pattern, lower, flags=re.IGNORECASE)
        if matches:
            signals.append(rule.name)
            reasons.append(rule.reason_tr)
            for item in matches[:6]:
                if isinstance(item, tuple):
                    item = " ".join(str(x) for x in item if x)
                matched_terms.append(str(item)[:80])
    return {
        "char_len": len(normalized),
        "word_count": len(normalized.split()) if normalized else 0,
        "signals": signals,
        "reasons": reasons,
        "top_salient_tokens": list(dict.fromkeys(matched_terms))[:12],
    }

