import torch
import torch.nn as nn


class CTCBeamSearchDecoder(nn.Module):
    def __init__(self, charset, blank_label=0, beam_size=10, cutoff_top_n=40, sparse=True):
        super().__init__()
        import ctcdecode
        self.decoder = ctcdecode.CTCBeamDecoder(
            charset,
            cutoff_top_n=cutoff_top_n,
            cutoff_prob=1.0,
            beam_width=beam_size,
            num_processes=16,
            blank_id=blank_label,
        )
        self.sparse = sparse
        self.cutoff_top_n = cutoff_top_n

    def forward(self, logits, logit_lengths=None):
        # T N C to N T C
        transposed_logits = torch.transpose(logits, 0, 1)
        probs = torch.softmax(transposed_logits, dim=-1)
        if self.sparse:
            topk_indices = torch.topk(probs, self.cutoff_top_n, dim=-1).indices
            n_indices = torch.arange(probs.size(0), device=topk_indices.device).view(-1, 1, 1).expand_as(topk_indices)
            t_indices = torch.arange(probs.size(1), device=topk_indices.device).view(1, -1, 1).expand_as(topk_indices)
            mask = t_indices < logit_lengths.view(-1, 1, 1).to(t_indices.device)
            valid_indices = torch.stack((n_indices, t_indices, topk_indices), dim=-1)
            valid_indices = valid_indices[mask].view(-1, 3)

            probs_sparse = probs[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
            N, T, _ = transposed_logits.shape
            beam_results, beam_scores, peak_frames, out_seq_lens = self.decoder.decode_sparse(probs_sparse, valid_indices, N, T, seq_lens=logit_lengths)

            # # for debug, verify the implementation of decode_sparse is equivalent to decode
            # _beam_results, _beam_scores, _peak_frames, _out_seq_lens = self.decoder.decode(probs, seq_lens=logit_lengths)
            # assert torch.all(_out_seq_lens == out_seq_lens), f'{_out_seq_lens} != {out_seq_lens}'
            # assert torch.all(_beam_scores == beam_scores), f'{_beam_scores} != {beam_scores}'
            # for i in range(beam_results.shape[0]):
            #     for j in range(beam_results.shape[1]):
            #         assert torch.all(_beam_results[i, j, :_out_seq_lens[i, j]] == beam_results[i, j, :_out_seq_lens[i, j]]), f'{_beam_results[i, j, :_out_seq_lens[i, j]]} != {beam_results[i, j, :_out_seq_lens[i, j]]}'
            #         assert torch.all(_peak_frames[i, j, :_out_seq_lens[i, j]] == peak_frames[i, j, :_out_seq_lens[i, j]]), f'{_peak_frames[i, j, :_out_seq_lens[i, j]]} != {peak_frames[i, j, :_out_seq_lens[i, j]]}'
        else:
            beam_results, beam_scores, peak_frames, out_seq_lens = self.decoder.decode(probs, seq_lens=logit_lengths)
        
        # beam_result.shape: BATCHSIZE x N_BEAMS X N_TIMESTEPS, out_seq_len.shape: BATCHSIZE x N_BEAMS
        return beam_results, beam_scores, peak_frames, out_seq_lens

