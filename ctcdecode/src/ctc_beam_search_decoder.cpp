#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

DecoderState::DecoderState(const std::vector<std::string> &vocabulary,
                           size_t beam_size,
                           double cutoff_prob,
                           size_t cutoff_top_n,
                           size_t blank_id,
                           int log_input,
                           Scorer *ext_scorer)
  : abs_time_step(0)
  , beam_size(beam_size)
  , cutoff_prob(cutoff_prob)
  , cutoff_top_n(cutoff_top_n)
  , blank_id(blank_id)
  , log_input(log_input)
  , vocabulary(vocabulary)
  , ext_scorer(ext_scorer)
{
  // assign space id
  auto it = std::find(vocabulary.begin(), vocabulary.end(), " ");
  // if no space in vocabulary
  if (it == vocabulary.end()) {
    space_id = -2;
  } else {
    space_id = std::distance(vocabulary.begin(), it);
  }

  // init prefixes' root
  root.score = root.log_prob_b_prev = 0.0;
  prefixes.push_back(&root);

  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
    fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
    root.set_dictionary(dict_ptr);
    auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
    root.set_matcher(matcher);
  }
}


void
DecoderState::next(const std::vector<std::vector<double>> &probs_seq)
{
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size(),
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step, ++abs_time_step) {
    auto &prob = probs_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(
          prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
      float blank_prob = log_input ? prob[blank_id] : std::log(prob[blank_id]);
      min_cutoff = prefixes[num_prefixes - 1]->score +
                   blank_prob - std::max(0.0, ext_scorer->beta);
      full_beam = (num_prefixes == beam_size);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n, log_input);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, abs_time_step, log_prob_c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            log_p += score;
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary


    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }

      prefixes.resize(beam_size);
    }
  }  // end of loop over time
}

void
DecoderState::next_sparse(const std::vector<std::vector<double>> &probs_seq,
                          const std::vector<std::vector<int>> &indices_seq)
{
  // dimension check
  size_t num_time_steps = probs_seq.size();

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step, ++abs_time_step) {
    auto &prob = probs_seq[time_step];
    auto &indice = indices_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    // assert ext_scorer is nullptr

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs_sparse(prob, indice, cutoff_prob, cutoff_top_n, log_input);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, abs_time_step, log_prob_c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            ngram = ext_scorer->make_ngram(prefix_to_score);
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            log_p += score;
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary


    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }

      prefixes.resize(beam_size);
    }
  }  // end of loop over time
}

std::vector<std::pair<double, Output>>
DecoderState::decode()
{
  std::vector<PathTrie*> prefixes_copy = prefixes;
  std::unordered_map<const PathTrie*, float> scores;
  for (PathTrie* prefix : prefixes_copy) {
    scores[prefix] = prefix->score;
  }

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    for (size_t i = 0; i < beam_size && i < prefixes_copy.size(); ++i) {
      auto prefix = prefixes_copy[i];
      if (!prefix->is_empty() && prefix->character != space_id) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
        score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
        score += ext_scorer->beta;
        scores[prefix] += score;
      }
    }
  }

  using namespace std::placeholders;
  size_t num_prefixes = std::min(prefixes_copy.size(), beam_size);
  std::sort(prefixes_copy.begin(), prefixes_copy.begin() + num_prefixes,
            std::bind(prefix_compare_external_scores, _1, _2, scores));

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes_copy.size(); ++i) {
    double approx_ctc = scores[prefixes_copy[i]];
    if (ext_scorer != nullptr) {
      std::vector<int> output;
      std::vector<int> timesteps;
      prefixes_copy[i]->get_path_vec(output, timesteps);
      auto prefix_length = output.size();
      auto words = ext_scorer->split_labels(output);
      // remove word insert
      approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
      // remove language model weight:
      approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
    }
    prefixes_copy[i]->approx_ctc = approx_ctc;
  }

  return get_beam_search_result(prefixes_copy, beam_size);
}

std::vector<std::pair<double, Output>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer)
{
  DecoderState state(vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id,
                     log_input, ext_scorer);
  state.next(probs_seq);
  return state.decode();
}

std::vector<std::pair<double, Output>> ctc_beam_search_decoder_sparse(
  const std::vector<std::vector<double>> &probs_seq,
  const std::vector<std::vector<int>> &indices_seq,
  const std::vector<std::string> &vocabulary,
  size_t beam_size,
  double cutoff_prob,
  size_t cutoff_top_n,
  size_t blank_id,
  int log_input,
  Scorer *ext_scorer)
{
DecoderState state(vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id,
                   log_input, ext_scorer);
state.next_sparse(probs_seq, indices_seq);
return state.decode();
}


std::vector<std::pair<double, Output>>  ctc_beam_search_decoder_with_given_state(
    const std::vector<std::vector<double>> &probs_seq,
    DecoderState *state,
    bool is_eos)
    {
      state->next(probs_seq);
      if (is_eos) {
        return state->decode();
      }
      else {
        return {};
      }

    }

std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer)
{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  std::cref(probs_split[i]),
                                  std::cref(vocabulary),
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  blank_id,
                                  log_input,
                                  ext_scorer));
  }



  // get decoding results
  std::vector<std::vector<std::pair<double, Output>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}


std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_sparse_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::vector<std::vector<int>>> &indices_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer)
{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder_sparse,
                                  std::cref(probs_split[i]),
                                  std::cref(indices_split[i]),
                                  std::cref(vocabulary),
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  blank_id,
                                  log_input,
                                  ext_scorer));
  }
  // get decoding results
  std::vector<std::vector<std::pair<double, Output>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}


std::vector<std::vector<std::pair<double, Output>>> ctc_beam_search_decoder_batch_with_states
(const std::vector<std::vector<std::vector<double>>> &probs_split,
    size_t num_processes,
    std::vector<void*> &states,
    const std::vector<bool> &is_eos_s)

{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();
  

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder_with_given_state,
                                  std::cref(probs_split[i]),
                                  static_cast<DecoderState*>(states[i]),
                                  is_eos_s[i]));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, Output>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
