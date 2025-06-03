import torch
from mamba_ssm.ops.triton.tree_scan import mask_index_gather
from einops import repeat
from mamba_ssm.utils.speculative_sampling import adjust_logits

# A specialized beam search verification function, where the acceptance criterion is to match exactly with beam search result
# Can build a general one with more flexible acceptance criterion.
def verify_beam_search_tree(input_tokens: torch.Tensor, 
                            input_mask: torch.Tensor, 
                            output_logits: torch.Tensor, 
                            log_probability: torch.Tensor,
                            sequence_cat: torch.Tensor,
                            base_num_beams: int, 
                            draft_num_beams: int):
    # input_tokens: num_beams x seqlen
    # output_logits: num_beam x seqlen x vocab_size
    actual_output = sequence_cat
    out_state_index = torch.arange(0, base_num_beams, device=input_tokens.device)
    output_prob = torch.softmax(output_logits, dim=-1)
    mask_sum = input_mask.sum(dim=-1)
    max_steps = int(torch.max(mask_sum).item())
    parent_matrix = mask_index_gather(input_mask, state_len=max_steps)
    curr_cond_prob = output_prob[:, 0, :]
    should_continue = True
    curr_step = 1
    correct_draft_last_token_index = None

    while (should_continue):
        # print(curr_cond_prob.shape, probability.shape)
        # Getting the current step actual output assuming that the previous step is already verify
        curr_step_log_prob = torch.log(curr_cond_prob) + log_probability # base_num_beam x vocab_size
        curr_step_log_prob = curr_step_log_prob.view((-1,))
        curr_step_topk = torch.topk(curr_step_log_prob, k=base_num_beams, dim=-1) # base_num_beam x base_num_beams
        # print(curr_step_prob.shape)
        # print(curr_step_topk)
        curr_value = curr_step_topk.values
        curr_index = curr_step_topk.indices
        # print(curr_index, output_logits.shape)
        state_index = (curr_index / output_logits.shape[-1]).to(torch.long)
        tokens = (curr_index % output_logits.shape[-1]).to(torch.long)
        # Now state_index and tokens hold the actual result, we can already put them in the actual output
        # change order if needed
        # print(actual_output, state_index)
        actual_output = actual_output[state_index]
        out_state_index = out_state_index[state_index]
        # concatenating the output
        actual_output = torch.cat([actual_output, tokens.view((base_num_beams, 1))], dim=1)
        # copy the new joint probability 
        log_probability.copy_(curr_value.view((base_num_beams, 1)))
        curr_step_actual = actual_output[:, -curr_step-1:] # base_num_beams x curr_step+1
        # print("in verification", log_probability.view((base_num_beams, 1)), curr_step_actual,curr_step, "25747:",  curr_step_log_prob[25747], torch.log(curr_cond_prob)[state_index, tokens])

        
        if (curr_step) < max_steps:
            # now we want to check whether the actual output of the current step matches any of the input
            # fetching the input tokens
            curr_step_input_index = torch.nonzero(mask_sum==curr_step+1)
            # print("mask sum,", mask_sum)
            # print(curr_step)
            # print("curr_step_input_index", curr_step_input_index)
            curr_step_input_state = curr_step_input_index[:, 0]
            curr_step_input_state_repeated = repeat(curr_step_input_state, 'b ... -> (b n) ...', n=curr_step+1)
            curr_parent_matrix = parent_matrix[mask_sum==curr_step+1]
            curr_parent_matrix = curr_parent_matrix[:, -curr_step-1:].reshape((-1, ))
            # print(curr_parent_matrix, curr_step_input_state)
            # print(input_tokens)
            # print("actual", curr_step_actual)
            curr_step_input_tokens = input_tokens[curr_step_input_state_repeated, curr_parent_matrix].view((curr_step_input_index.shape[0], -1)) # draft_num_beam x curr_step+1
            # print("input",curr_step_input_tokens)

            # comparing the state indexes as even a token match from a different beam shouldn't be accepted
            state_comparison = curr_step_input_state[None, :] == out_state_index[:, None]

            # comparing the tokens
            token_comparison = curr_step_input_tokens[None, :, :] == curr_step_actual[:, None, :] # base_num_beams x draft_num_beams x curr_step+1
            token_comparison = torch.prod(token_comparison, dim=-1) # base_num_beams x draft_num_beams
            # We need the input to catch all the actual output
            comparison = state_comparison * token_comparison
            criterion = torch.sum(comparison, dim=1)
            # print(curr_step+1, max_steps)
            # print("criterion", criterion)
            # need to handle the case where two beams has the exact same input, in that case criterion will be equal to 2 and there will be more than 
            # 1 ones in each column of comparison
            if (criterion == 1).all():
                # If this step is successful we continue to the next step
                # We need to update the curr_cond_prob to reflect the correct token
                # Find out which drafted_beam is correct 
                correct_index_pair = torch.nonzero(comparison) # tell us which actual seq match which draft seq TODO: What if two beams has the same input?
                correct_draft_state = curr_step_input_index[correct_index_pair[:,1], torch.zeros_like(correct_index_pair[:,1])]
                correct_draft_last_token_index = curr_step_input_index[correct_index_pair[:,1], -1]
                curr_cond_prob = output_prob[correct_draft_state, correct_draft_last_token_index, :]
                # print("prob tokens", input_tokens[correct_draft_state, correct_draft_last_token_index])
                # reordering base on the match index. 
                curr_cond_prob = curr_cond_prob[torch.argsort(correct_index_pair[:,0])]
                should_continue = True
                curr_step += 1
            else:
                # If this step is unsuccessfull, we don't really have to do anything as the actual output is already recorded
                should_continue = False
        else:
            # correct_draft_last_token_index = correct_draft_last_token_index[state_index] 
            should_continue = False
    
    if correct_draft_last_token_index is None:
        correct_draft_last_token_index = torch.zeros((base_num_beams,), device=input_mask.device, dtype=torch.long)
    else:
        # print("state index in exit", state_index)
        correct_draft_last_token_index = correct_draft_last_token_index[state_index] # re-ordering the output mask if necessary since the last topk may change the order
    verified_mask = input_mask[out_state_index, correct_draft_last_token_index, :]
    # print(actual_output, out_state_index, probability, curr_step, verified_mask)
    # print(verified_mask)
    # print(out_state_index)
    return actual_output, out_state_index, log_probability, curr_step, verified_mask


def sampling_verification_tree(input_tokens: torch.Tensor, 
                                input_mask: torch.Tensor, 
                                output_logits: torch.Tensor, 
                                log_probability: torch.Tensor,
                                sequence_cat: torch.Tensor,
                                top_k=1,
                                temperature=1,
                                top_p=0,
                                min_p=0,
                                generator=None):
    assert input_tokens.shape[0] == 1 # TODO: assuming there is only 1 sequence
    adjusted_output_logits = adjust_logits(output_logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
    actual_output = sequence_cat
    should_continue = True
    curr_step = 1
    adjusted_output_prob = torch.softmax(adjusted_output_logits, dim=-1)
    mask_sum = input_mask.sum(dim=-1)
    max_steps = int(torch.max(mask_sum).item())
    curr_cond_prob = adjusted_output_prob[:, 0, :]
    parent_matrix = mask_index_gather(input_mask, state_len=max_steps)
    last_correct_token_index = torch.zeros(1, dtype=torch.long, device=input_mask.device)

    while should_continue:
        # sample the token base on the current conditional probability
        curr_token = torch.multinomial(curr_cond_prob, 1, generator=generator) # shape: (1, 1)
        actual_output = torch.cat([actual_output, curr_token], dim=1)

        if curr_step < max_steps:
            
            # not only do we want the child at the next step, we also want the child of the correct token
            curr_step_input_index = torch.nonzero(torch.logical_and(mask_sum==curr_step+1, parent_matrix[0,:,-2]==last_correct_token_index))
            curr_child_token = input_tokens[curr_step_input_index[:,0], curr_step_input_index[:,1]]

            comparison = curr_token == curr_child_token
            # If the current generated token for a tree node match with any child node, we can continue
            if comparison.any():
                correct_index = torch.nonzero(comparison)[0, 1] # there should be only 1 match
                last_correct_token_index = curr_step_input_index[correct_index,1]
                curr_cond_prob = adjusted_output_prob[[0], last_correct_token_index, :]
                curr_step += 1
            else:
                should_continue = False
        else:
            should_continue = False
    
    verified_mask = input_mask[[0], last_correct_token_index, :]
    out_state_index = torch.zeros((1,), dtype=torch.long, device=input_mask.device)
    return actual_output, out_state_index, log_probability, curr_step, verified_mask


def unroll_tree(input_tokens: torch.Tensor, 
                input_mask: torch.Tensor):
    '''
    Take a tree sequence and unroll it into multiple sequence with common prefixes.
    returns a tensor with (num_leaves, tree_depth) dimension  
    '''

    assert input_tokens.shape[0] == 1 # TODO: assuming there is only 1 batch in the input tree
    mask_sum = torch.sum(input_mask, dim=-1)
    max_steps = int(torch.max(mask_sum).item())
    parent_matrix = mask_index_gather(input_mask, state_len=max_steps)
    parent_node_indices = torch.unique(parent_matrix[:, :, -2])
    # print(parent_node_indices)
    # print(parent_matrix)
    all_node_indices = torch.arange(input_tokens.shape[1], device=input_tokens.device)
    # print(~torch.isin(all_node_indices, parent_node_indices))
    leaf_node_indices = all_node_indices[~torch.isin(all_node_indices, parent_node_indices)]
    # print(leaf_node_indices)

    leaf_node_matrix = parent_matrix[:, leaf_node_indices, :]
    leaf_node_mask = leaf_node_matrix != -1
    # print(input_tokens, leaf_node_matrix)
    leaf_node_matrix_out = leaf_node_matrix.clone()
    leaf_node_matrix = torch.where(leaf_node_mask, leaf_node_matrix, 0)
    leaf_node_sequences = torch.gather(repeat(input_tokens, "b l -> (b n) l", n=leaf_node_matrix.shape[1]), 1, leaf_node_matrix[0])
    
    return leaf_node_sequences, leaf_node_mask[0], leaf_node_matrix_out


def compress_tree(output_logits: torch.Tensor,
                  leaf_node_matrix: torch.Tensor):
    '''
    Compressing multiple batches of output logits into a prefix tree of batch size 1
    Use to convert the unrolled output logits back to the tree sequence.
    '''

    num_tokens = torch.max(leaf_node_matrix) + 1
    output_tree_logit = torch.empty((1, num_tokens, output_logits.shape[2]), device=output_logits.device)
    filled_mask = torch.zeros((1, num_tokens))
    leaf_node_matrix = leaf_node_matrix.view((output_logits.shape[0]*output_logits.shape[1]))
    output_logits = output_logits.view((output_logits.shape[0]*output_logits.shape[1], output_logits.shape[2]))
    for i in range(output_logits.shape[0]):
        if leaf_node_matrix[i] == -1:
            continue
        pos = leaf_node_matrix[i]
        if filled_mask[0, pos] == 1:
            continue
        output_tree_logit[:, pos, :] = output_logits[i, :]
        filled_mask[0, pos] = 1
    
    return output_tree_logit

class TokenTreeNode:
    def __init__(self, 
                 parent_index,
                 children_node_indices,
                 map_index
                 ):
        self.parent_index = parent_index
        self.children = children_node_indices
        self.map_index = map_index
    
    def __repr__(self):
        return "<tree node index={}, parent_index={}, num_children={}>".format(self.map_index, self.parent_index, len(self.children))

class TokenTree:
    def __init__(self, 
                 input_token,
                 input_mask,
                 target_logits, 
                 draft_logits):
        # initialzing an empty tree
        if input_token == None:
            self.input_mask = None
            self.target_logits = None
            self.input_token = None
            self.draft_logits = None
            self.root = None
            self.token_to_sequence_map = None
            self.num_seq = 0
        else:
            self.input_mask = input_mask
            self.target_logits = target_logits
            self.input_token = input_token
            self.draft_logits = draft_logits

            self.update_parent_matrix()
            # self.target_prob = torch.softmax(target_logits, dim=-1)
            # self.draft_prob = torch.softmax(draft_logits, dim=-1)
            self.root = self.create_index_map_from_mask(torch.scalar_tensor(0, device=target_logits.device, dtype=torch.long), None)
    
    def update_parent_matrix(self):
        # variables to assist in tree creation
        self.mask_sum = self.input_mask.sum(dim=-1)
        self.max_steps = int(torch.max(self.mask_sum).item())
        self.num_tokens = self.input_mask.shape[1]
        self.parent_matrix = mask_index_gather(self.input_mask, state_len=self.max_steps)
    
    def create_index_map_from_mask(self, node_index, parent):
        parent_indices = self.parent_matrix[:, :, -2]
        child_index = torch.nonzero(parent_indices==node_index)[:,1]
        
        root_node = TokenTreeNode(parent, [], node_index)
        if child_index.shape[0] == 0:
            return root_node
        child_nodes = []
        for i in child_index:
            child_node = self.create_index_map_from_mask(i, node_index)
            child_nodes.append(child_node)
        
        root_node.children = child_nodes

        return root_node

    def add_sequence_to_tree(self, sequence, draft_logits, target_logits):
        '''
        sequence: (1, S)
        draft_logit: (1, S, VOCAB_SIZE)
        target_logit: (1, S, VOCAB_SIZE)
        '''
        parent_node = None
        # If there is no tree at the start
        if self.root is None:
            self.input_token = sequence
            self.input_mask = torch.tril(torch.ones((sequence.shape[1], sequence.shape[1]), device=sequence.device, dtype=torch.long)).unsqueeze(0)
            self.target_logits = target_logits
            self.draft_logits = draft_logits
            self.update_parent_matrix()
            
            self.root = self.create_index_map_from_mask(torch.scalar_tensor(0, device=sequence.device, dtype=torch.long), None)
            self.token_to_sequence_map.update({i:self.num_seq for i in range(sequence.shape[1])})
            self.num_seq += 1
            return

        parent_node = None
        curr_child_index = [self.root]
        for s in range(sequence.shape[1]):
            found = False
            for curr in curr_child_index:
                curr_token = self.input_token[:, curr.map_index]
                if sequence[:, s] == curr_token:
                    curr_child_index = curr.children
                    parent_node = curr
                    found = True
                    break
            
            if not found:
                self.input_token = torch.cat([self.input_token, sequence[:, [s]]], dim=1)
                self.input_mask = torch.nn.functional.pad(self.input_mask, (0, 1, 0, 1), value=0)
                self.input_mask[:, self.input_token.shape[1]-1, :] = self.input_mask[:, parent_node.map_index, :]
                self.input_mask[:, self.input_token.shape[1]-1, self.input_token.shape[1]-1] = 1
                if self.draft_logits is not None:
                    self.draft_logits = torch.cat([self.draft_logits, draft_logits[:, [s], :]], dim=1)
                if self.target_logits is not None:
                    self.target_logits = torch.cat([self.target_logits, target_logits[:, [s], :]], dim=1)
                new_node = TokenTreeNode(parent_node.map_index, [], self.input_token.shape[1]-1)
                curr_child_index.append(new_node)
                parent_node = new_node
                curr_child_index = new_node.children
                self.token_to_sequence_map[self.input_token.shape[1]-1] = self.num_seq
        
        self.num_seq += 1 
        self.update_parent_matrix()

    def multi_step_speculative_sampling(self, 
                                        sequence_cat, 
                                        log_probability,
                                        top_k=1,
                                        temperature=1,
                                        top_p=0,
                                        min_p=0,
                                        generator=None):
        '''
        Performs multi_step_speculative (MSS) sampling detailed in SpecInfer (Miao et. al.)
        Referencing implementation from SpecExec: https://github.com/yandex-research/specexec/blob/main/specdec/spec_SI.py
        '''
        adjusted_target_logits = adjust_logits(self.target_logits, top_k, top_p, min_p, temperature)
        adjusted_draft_logits = adjust_logits(self.draft_logits, top_k, top_p, min_p, temperature)
        self.target_prob = torch.softmax(adjusted_target_logits, dim=-1)
        self.draft_prob = torch.softmax(adjusted_draft_logits, dim=-1)

        curr_node = self.root
        curr_node_index = curr_node.map_index
        target_dist_adj = self.target_prob[:, curr_node_index, :]
        should_continue = True
        actual_output = sequence_cat
        curr_step = 1
        while should_continue:
            children = curr_node.children
            curr_node_index = curr_node.map_index
            target_dist_adj = self.target_prob[:, curr_node_index, :]
            if len(children) > 0:
            # at this point there is at least 1 child in for the curr token
                draft_dist_adj = self.draft_prob[:, children[0].map_index, :]
                match_found = False
                for child in children:
                    token_id = self.input_token[:, child.map_index]
                    target_token_prob = target_dist_adj[:, token_id]
                    draft_token_prob = draft_dist_adj[:, token_id]
                    if draft_token_prob == 0.0: # when topk tokens ended up outside the topp tokens, this will happen,
                        continue
                    p_accept = (target_token_prob / draft_token_prob).clamp_max(1.0)
                    r = torch.rand(1, device=self.target_prob.device, generator=generator)
                    # print("curr: {}, {}, {}".format(curr_node_index, child.map_index, token_id))
                    # print(p_accept, r)
                    if r <= p_accept:
                        # accepted
                        actual_output = torch.cat([actual_output, token_id.unsqueeze(0)], dim=1)
                        curr_step += 1
                        curr_node = child
                        match_found = True
                        break
                    else:
                        # rejected one child, we adjust the target and draft distribution
                        target_dist_adj = (target_dist_adj - draft_dist_adj).clamp_min(0.0)
                        target_dist_adj = torch.nn.functional.normalize(target_dist_adj, dim=1, p=1)

                        draft_dist_adj[:, token_id] = 0
                        draft_dist_adj = torch.nn.functional.normalize(draft_dist_adj, dim=1, p=1)
                    # print(target_dist_adj, draft_dist_adj)
            # print('match found', match_found)
            if not match_found or curr_step >= self.max_steps or len(children) == 0:
                # not match found in children, we sample with the final adjusted dist and terminate
                # print(target_dist_adj)
                should_continue = False
                if not match_found:
                    sampled_token = torch.multinomial(target_dist_adj, 1, generator=generator)
                    actual_output = torch.cat([actual_output, sampled_token], dim=1)
                else:
                    # case where match is found but we reach the end of the draft
                    # In this case we just sample from the curr_node
                    sampled_token = torch.multinomial(self.target_prob[:, curr_node.map_index, :], 1)
                    actual_output = torch.cat([actual_output, sampled_token], dim=1)

        out_state_index = torch.zeros((1,), dtype=torch.long, device=self.target_prob.device)
        verified_mask = self.input_mask[:, curr_node.map_index, :]
        return actual_output, out_state_index, log_probability, curr_step, verified_mask
    
    @staticmethod
    def from_independent_sequences(input_token, 
                                   input_mask,
                                   draft_logits=None,
                                   target_logits=None):
        tree = TokenTree(input_token=None, input_mask=None, target_logits=None, draft_logits=None)
        bool_mask = input_mask.to(torch.bool)
        tree.token_to_sequence_map = {}
        for b in range(input_token.shape[0]):
            seq = torch.masked_select(input_token[[b], :], bool_mask[[b], :]).unsqueeze(0)
            if draft_logits is not None:
                dl = torch.masked_select(draft_logits[[b], :, :], bool_mask[[b], :, None]).view((1, -1, draft_logits.shape[-1]))
            else:
                dl = None
            if target_logits is not None:
                tl = torch.masked_select(target_logits[[b], :, :], bool_mask[[b], :, None]).view((1, -1, target_logits.shape[-1]))
            else:
                tl = None
            tree.add_sequence_to_tree(seq, dl, tl)
        
        return tree

    def get_token_sequence(self, verified_mask):
        pos = torch.max(torch.arange(verified_mask.shape[1], device=verified_mask.device) * verified_mask, dim=1).values
        return self.token_to_sequence_map[pos.item()]
