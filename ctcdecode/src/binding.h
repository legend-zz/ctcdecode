int paddle_beam_decode(THFloatTensor *th_probs,
                       THIntTensor *th_seq_lens,
                       std::vector<std::string> labels,
                       int vocab_size,
                       size_t beam_size,
                       size_t num_processes,
                       double cutoff_prob,
                       size_t cutoff_top_n,
                       size_t blank_id,
                       int log_input,
                       THIntTensor *th_output,
                       THIntTensor *th_timesteps,
                       THFloatTensor *th_scores,
                       THIntTensor *th_out_length);

int paddle_beam_decode_sparse(THFloatTensor *th_probs_sparse,
                              THIntTensor *th_indices,
                              size_t batch_size,
                              size_t max_seq_len,
                              THIntTensor *th_seq_lens,
                              std::vector<std::string> labels,
                              int vocab_size,
                              size_t beam_size,
                              size_t num_processes,
                              double cutoff_prob,
                              size_t cutoff_top_n,
                              size_t blank_id,
                              int log_input,
                              THIntTensor *th_output,
                              THIntTensor *th_timesteps,
                              THFloatTensor *th_scores,
                              THIntTensor *th_out_length);

int paddle_beam_decode_lm(THFloatTensor *th_probs,
                          THIntTensor *th_seq_lens,
                          std::vector<std::string> labels,
                          int vocab_size,
                          size_t beam_size,
                          size_t num_processes,
                          double cutoff_prob,
                          size_t cutoff_top_n,
                          size_t blank_id,
                          bool log_input,
                          int *scorer,
                          THIntTensor *th_output,
                          THIntTensor *th_timesteps,
                          THFloatTensor *th_scores,
                          THIntTensor *th_out_length);

void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        std::vector<std::string> labels,
                        int vocab_size);


void* paddle_get_decoder_state(const std::vector<std::string> &vocabulary,
                               size_t beam_size,
                               double cutoff_prob,
                               size_t cutoff_top_n,
                               size_t blank_id,
                               int log_input,
                               void* scorer);

void paddle_release_scorer(void* scorer);
void paddle_release_state(void* state);


int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);
