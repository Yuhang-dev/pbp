from __future__ import annotations

from pbp.data import split_hh_rlhf_pair


def test_split_hh_rlhf_pair_uses_final_assistant_turn_as_prompt_boundary():
    chosen = "\n\nHuman: hello\n\nAssistant: helpful answer"
    rejected = "\n\nHuman: hello\n\nAssistant: unhelpful answer"

    prompt, chosen_response, rejected_response = split_hh_rlhf_pair(chosen, rejected)

    assert prompt == "\n\nHuman: hello\n\nAssistant:"
    assert chosen_response == " helpful answer"
    assert rejected_response == " unhelpful answer"
