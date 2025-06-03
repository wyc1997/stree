import torch
from mamba_ssm.utils.generation_utils import modify_logits_for_top_k_filtering, modify_logits_for_min_p_filtering, modify_logits_for_top_p_filtering, modify_logit_for_repetition_penalty

def adjust_logits(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    if top_k == 1:  # Short-circuit for greedy decoding
        return torch.where(logits == logits.max(dim=-1,keepdim=True).values, 1.0, -torch.inf).to(dtype=logits.dtype, device=logits.device)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            modify_logits_for_top_k_filtering(logits, top_k=top_k)
            if temperature != 1.0:
                logits_top /= temperature
                logits_top = logits_top - torch.max(logits_top, dim=-1, keepdim=True).values
            modify_logits_for_top_p_filtering(logits, top_p)
            return logits
        else:
            if min_p > 0.0:
                logits_top = logits.clone()
                max_prob = logits_top[..., 0].item()
                min_prob = max_prob * min_p
                modify_logits_for_min_p_filtering(logits_top, min_prob)
                if temperature != 1.0:
                    logits_top /= temperature
                    logits_top = logits_top - torch.max(logits_top, dim=-1, keepdim=True).values
                return logits_top
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            logits_top = logits_top - torch.max(logits_top, dim=-1, keepdim=True).values
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return logits_top


# inspired by https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/speculative_sampling.py
# TODO: not tested yet. 
def speculative_sampling(draft_logits, draft_token, verification_logits, temperature=1.0, top_k=1, top_p=0.0, min_p=0.0, generator=None):
    '''
    draft_logits: (batch, seqlen, vocab_size)
    draft_token: (batch, seqlen)
    verfication_logits: (batch, seqlen + 1, vocab_size)
    Implementing speculative sampling by Leviathan et al. Specifcally, we do: 
    1. Sample from the adjusted draft logits
    2. Generate r ~ uniform(0,1)
    3. if r > veri_prob / draft_prob: we accept the sampled logits
    4. else: we sample again from normalize(veri_prob - draft_prob) at that position where we make a mistake. 

    Note: works correctly on ndraft=1. On Ndraft=2 (or anything >1), it will run but the distribution sampled will be different from
    original. This is because doing speculative sampling twice on two independent sentence actually gives a higher prob of accepting 
    token with low prob
    '''
    # adjusting logits
    adjusted_draft_logits = adjust_logits(draft_logits, top_k, top_p, min_p, temperature)
    adjusted_verification_logits = adjust_logits(verification_logits, top_k, top_p, min_p, temperature)
    
    # Getting probability 
    adjusted_draft_prob = torch.softmax(adjusted_draft_logits, dim=-1)
    adjusted_verification_prob = torch.softmax(adjusted_verification_logits, dim=-1)

    acceptance_threshold = (adjusted_verification_prob[:, :-1, :] / adjusted_draft_prob).clamp_max(1.0)
    acceptance_threshold = acceptance_threshold.view((-1, acceptance_threshold.shape[2]))
    flattened_draft_token = draft_token.view((-1))
    acceptance_threshold = acceptance_threshold[torch.arange(acceptance_threshold.shape[0]), flattened_draft_token].view(draft_logits.shape[:2])
    r = torch.rand_like(draft_logits[:,:,0])

    # print(r, acceptance_threshold)
    left_contig = torch.cumprod(r <= acceptance_threshold, dim=1)
    num_accept, batch_num = torch.max(torch.sum(left_contig, dim=1), dim=0)

    new_prob = torch.zeros_like(adjusted_verification_prob)
    new_prob[:, :-1, :] = (adjusted_verification_prob[:, :-1, :] - adjusted_draft_prob).clamp_min(0.0)
    new_prob[:, -1, :] = adjusted_verification_prob[:, -1, :]
    # normalization
    new_prob = torch.nn.functional.normalize(new_prob, dim=-1, p=1)
    next_sample_prob = new_prob[[batch_num], num_accept, :]
    next_sample = torch.multinomial(next_sample_prob, 1, generator=generator)

    # TODO: This will work for batch size = 1 only 
    return torch.cat([draft_token[[batch_num],:num_accept], next_sample], dim=1), batch_num


def sampling_verification(draft_token, verification_logits, temperature=1.0, top_k=1, top_p=0.0, min_p=0.0, generator=None):
    '''
    This function samples from the verification logits one by one and then verify whether the sampled token 
    is the same as the drafted token. If yes, we can procceed to the next token 
    '''
    assert verification_logits.shape[0] == 1 # TODO: assumes there is only one sequence
    adjusted_verification_logits = adjust_logits(verification_logits, top_k, top_p, min_p, temperature)
    adjusted_verification_prob = torch.softmax(adjusted_verification_logits, dim=-1)
    # iterating over the sequence dimension to sample one by one
    output = []
    for i in range(verification_logits.shape[1]-1):
        sampled_token = torch.multinomial(adjusted_verification_prob[:, i, :], 1, generator=generator)
        output.append(sampled_token)
        drafted_token = draft_token[:,i+1]
        if (drafted_token != sampled_token).all():
            break
    # everything is correct so we want to sample the last token as well
    if len(output) == verification_logits.shape[1]-1:
        sampled_token = torch.multinomial(adjusted_verification_prob[:, -1, :], 1, generator=generator)
        output.append(sampled_token)

    return torch.cat(output, dim=1)







