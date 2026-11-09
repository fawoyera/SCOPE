"""Prompt construction.  The same prompt is used by every method for a given model,
so differences in the results table come from the method, not the prompt."""
from __future__ import annotations

from typing import Dict, Optional

SYSTEM_PROMPTS = {
    "atc": (
        "You are an expert in Air Traffic Control (ATC) radio communication "
        "for Unmanned Aerial Vehicle (UAV) operations in controlled airspace.\n\n"
        "Your task is to generate the RESPONSE side of an ATC radio exchange, "
        "given the INPUT side. The response must conform strictly to "
        "ICAO Doc 4444 Chapter 12 standard phraseology.\n\n"
        "MANDATORY RULES:\n"
        "1. INDIVIDUAL DIGIT PRONUNCIATION — speak each digit separately:\n"
        "   correct: ONE EIGHT ZERO  |  wrong: ONE EIGHTY  or  180\n"
        "   correct: NINER THOUSAND  |  wrong: NINE THOUSAND\n"
        "   correct: ONE ONE EIGHT DECIMAL THREE  |  wrong: ONE-ONE-EIGHT-POINT-THREE\n"
        "2. NO DISFLUENCIES — omit all filler words:\n"
        "   forbidden: AH, UH, UM, OK, OKAY, RIGHT, YEAH, SURE, SO, WELL\n"
        "3. TERSE REGISTER — use the minimum words required by procedure:\n"
        "   correct: DELTA SIX TURN LEFT HEADING ONE EIGHT ZERO\n"
        "   wrong:   DELTA SIX PLEASE TURN LEFT TO A HEADING OF ONE EIGHTY DEGREES\n"
        "4. CALLSIGN REQUIRED — include the aircraft callsign where procedure demands:\n"
        "   controller addressing pilot: begin with callsign (DELTA SIX DESCEND AND MAINTAIN...)\n"
        "   pilot readback: include callsign (DESCEND AND MAINTAIN THREE THOUSAND DELTA SIX)\n"
        "5. STANDARD WORDS — use only ICAO-prescribed vocabulary:\n"
        "   AFFIRM not AFFIRMATIVE | NINER not NINE | WILCO not WILL DO\n"
        "   ROGER for acknowledgement | CORRECTION when self-correcting"
    ),
    "smcp": (
        "You are an expert in maritime radio communication for vessel operations.\n\n"
        "Your task is to generate the RESPONSE side of a maritime radio exchange, "
        "given the INPUT side. The response must conform strictly to "
        "IMO SMCP (Standard Marine Communication Phrases) standards.\n\n"
        "MANDATORY RULES:\n"
        "1. Use standard IMO SMCP phrases — do not paraphrase.\n"
        "2. Be terse — maritime radio uses minimum necessary words.\n"
        "3. Include vessel name/callsign where required by procedure.\n"
        "4. No disfluencies: AH, UM, OK, OKAY are forbidden."
    ),
}

PREAMBLE = {
    "atc": ("Below is a radio communication exchange in controlled airspace. "
            "Complete the response side following strict ICAO phraseology standards."),
    "smcp": ("Below is a maritime radio communication exchange. "
             "Complete the response side following strict IMO SMCP standards."),
}

EXCHANGE_TYPE_LABELS = {
    "instruction_readback": "ATC instruction followed by pilot readback",
    "pilot_to_controller": "Pilot report or request followed by ATC response",
    "handoff": "Frequency handoff or acknowledgement exchange",
    "other": "General ATC/Pilot radio exchange",
}

ROLE_LABELS = {
    "controller": "Air Traffic Controller (ATC)",
    "pilot": "UAV Pilot",
    "unified": "ATC or Pilot",
}


def select_target(pair: dict) -> str:
    """Training/eval target. gold: expert > judge; silver: judge; bronze: raw transcript."""
    tier = pair.get("quality_tier", "bronze")
    if tier == "gold":
        t = pair.get("expert_corrected") or pair.get("judge_corrected")
        if t and t.strip():
            return t.strip()
    if tier in ("gold", "silver"):
        t = pair.get("judge_corrected")
        if t and t.strip():
            return t.strip()
    return (pair.get("resp_text") or pair.get("resp_text_raw") or "").strip()


# Prompt explicitness, from least to most informative (ablation axis):
#   minimal  – the incoming transmission only; no task description, roles, context or rules
#   task     – task description + exchange context + roles; no phraseology rules
#   rules    – task + the explicit ICAO phraseology rules (system prompt)   ← default, used in the paper
#   fewshot  – rules + k fixed gold demonstrations from the training split (same response role)
PROMPT_STYLES = ("minimal", "task", "rules", "fewshot")


def format_demonstrations(demos, req_label: str, resp_label: str) -> str:
    return "\n\n".join(f"Input ({req_label}): {d['req_text'].strip()}\nResponse ({resp_label}): {select_target(d)}"
                        for d in demos)


def build_prompt(pair: dict, tokenizer=None, prompt_format: str = "alpaca",
                 system_key: str = "atc", chat_template_kwargs: Optional[dict] = None,
                 merge_system_into_user: bool = False, style: str = "rules",
                 demonstrations: Optional[list] = None) -> Dict[str, str]:
    if style not in PROMPT_STYLES:
        raise ValueError(f"prompt style must be one of {PROMPT_STYLES}, got {style!r}")
    req_text = (pair.get("req_text") or pair.get("request") or "").strip()
    req_role = pair.get("req_role", "unknown")
    resp_role = pair.get("resp_role", "unified")
    req_label = ROLE_LABELS.get(req_role, str(req_role).title())
    resp_label = ROLE_LABELS.get(resp_role, str(resp_role).title())
    system = SYSTEM_PROMPTS[system_key]
    preamble = PREAMBLE.get(system_key, PREAMBLE["atc"])

    ctx = [f"Exchange type: {EXCHANGE_TYPE_LABELS.get(pair.get('exchange_type', 'other'), pair.get('exchange_type', 'other'))}"]
    if pair.get("airport"):
        ctx.append(f"Airport / facility: {pair['airport']}")
    if pair.get("callsign"):
        ctx.append(f"Aircraft callsign: {pair['callsign']}")
    ctx.append(f"Generate the response for: {resp_label}")
    context = "\n".join(ctx)
    demos = ""
    if style == "fewshot" and demonstrations:
        demos = "### Examples:\n" + format_demonstrations(demonstrations, req_label, resp_label) + "\n\n"

    if style != "rules" or demos:
        return _build_styled(style, prompt_format, tokenizer, chat_template_kwargs, merge_system_into_user,
                             system, preamble, context, demos, req_text, req_label, resp_label, pair, resp_role)

    if prompt_format == "chat":
        if tokenizer is None:
            raise ValueError("prompt_format=chat needs a tokenizer")
        user = f"{preamble}\n\n### Context:\n{context}\n\n### Input ({req_label}):\n{req_text}"
        msgs = ([{"role": "user", "content": f"{system}\n\n{user}"}] if merge_system_into_user
                else [{"role": "system", "content": system}, {"role": "user", "content": user}])
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               **(chat_template_kwargs or {}))
        sep = ""
    elif prompt_format == "alpaca":
        prompt = (f"{preamble}\n\n### System:\n{system}\n\n### Context:\n{context}\n\n"
                  f"### Input ({req_label}):\n{req_text}\n\n### Response ({resp_label}):")
        sep = " "
    else:
        raise ValueError(f"Unknown prompt_format {prompt_format!r}")

    return {"prompt": prompt, "sep": sep, "target": select_target(pair),
            "request": req_text, "role": resp_role}


def _build_styled(style, prompt_format, tokenizer, chat_kwargs, merge_system, system, preamble, context,
                  demos, req_text, req_label, resp_label, pair, resp_role) -> Dict[str, str]:
    """Non-default explicitness levels.  ``rules`` (above) is kept byte-identical to the original prompt."""
    use_system = style in ("rules", "fewshot")
    if style == "minimal":
        body, header = f"### Input:\n{req_text}", "### Response:"
    else:
        body = f"{preamble}\n\n### Context:\n{context}\n\n{demos}### Input ({req_label}):\n{req_text}"
        header = f"### Response ({resp_label}):"
    if prompt_format == "chat":
        if tokenizer is None:
            raise ValueError("prompt_format=chat needs a tokenizer")
        if use_system:
            msgs = ([{"role": "user", "content": f"{system}\n\n{body}"}] if merge_system
                    else [{"role": "system", "content": system}, {"role": "user", "content": body}])
        else:
            msgs = [{"role": "user", "content": body}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **(chat_kwargs or {}))
        sep = ""
    elif prompt_format == "alpaca":
        if use_system:
            prompt = f"{preamble}\n\n### System:\n{system}\n\n" + body.split("\n\n", 1)[1] + f"\n\n{header}"
        else:
            prompt = f"{body}\n\n{header}"
        sep = " "
    else:
        raise ValueError(f"Unknown prompt_format {prompt_format!r}")
    return {"prompt": prompt, "sep": sep, "target": select_target(pair), "request": req_text, "role": resp_role}
