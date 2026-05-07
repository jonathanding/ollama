# Graph Report - llama/ + server/ + llm/ + model/  (2026-04-07)

## Corpus Check
- 411 files · ~954,393 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6552 nodes · 13876 edges · 144 communities detected
- Extraction: 48% EXTRACTED · 52% INFERRED · 0% AMBIGUOUS · INFERRED: 7217 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `ma_free()` - 144 edges
2. `ma_malloc()` - 82 edges
3. `ma_get_bytes_per_frame()` - 75 edges
4. `ma_log_postf()` - 60 edges
5. `ma_device_get_log()` - 54 edges
6. `ma_context_get_log()` - 46 edges
7. `ma_log_post()` - 44 edges
8. `ma_spatializer_process_pcm_frames()` - 38 edges
9. `stbi__get8()` - 38 edges
10. `ma_strncpy_s()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `New()` --calls--> `newMultiModalProjector()`  [INFERRED]
  model\models\qwen3vl\model.go → model\models\mistral3\model.go
- `New()` --calls--> `inferRecurrentLayers()`  [INFERRED]
  model\models\qwen3vl\model.go → model\models\qwen3next\model.go
- `New()` --calls--> `defaultVHeadReordered()`  [INFERRED]
  model\models\qwen3vl\model.go → model\models\qwen3next\model.go
- `llama_kv_cache()` --calls--> `size_k_bytes()`  [INFERRED]
  llama\llama.cpp\src\llama-kv-cache.h → llama\llama.cpp\src\llama-kv-cache.cpp
- `llama_kv_cache()` --calls--> `size_v_bytes()`  [INFERRED]
  llama\llama.cpp\src\llama-kv-cache.h → llama\llama.cpp\src\llama-kv-cache.cpp

## Hyperedges (group relationships)
- **llama.cpp Vendoring System** — readme_llama_package, readme_llama_cpp, readme_ggml, readme_makefile_sync, readme_vendor_directory, readme_patch_workflow, readme_fetch_head [EXTRACTED 1.00]
- **Ollama Registry Model Format** — registry_smol_manifest, registry_docker_manifest_v2, registry_ollama_model_mediatype, registry_gguf_blob [EXTRACTED 1.00]

## Communities

### Community 0 - "Miniaudio Async/Events"
Cohesion: 0.0
Nodes (363): ma_async_notification_event__on_signal(), ma_async_notification_event_signal(), ma_async_notification_poll_init(), ma_async_notification_poll_is_signalled(), ma_atomic_compare_and_swap_16(), ma_atomic_compare_and_swap_8(), ma_atomic_compare_exchange_strong_explicit_16(), ma_atomic_compare_exchange_strong_explicit_8() (+355 more)

### Community 1 - "Miniaudio Memory/Atomics"
Cohesion: 0.01
Nodes (374): ma_aligned_free(), ma_aligned_malloc(), ma_allocation_callbacks_init_copy(), ma_allocation_callbacks_init_default(), ma_atomic_vec3f_init(), ma_attenuation_exponential(), ma_audio_buffer_alloc_and_init(), ma_audio_buffer_init() (+366 more)

### Community 2 - "Miniaudio Platform Audio"
Cohesion: 0.01
Nodes (366): ma_add_native_data_format_to_device_info_from_WAVEFORMATEX(), ma_allocate_AudioBufferList__coreaudio(), ma_audio_buffer_ref__data_source_on_get_data_format(), ma_audio_worklet_process_callback__webaudio(), ma_best_format_from_fd__audio4(), ma_buffer_queue_callback_capture__opensl_android(), ma_buffer_queue_callback_playback__opensl_android(), ma_calculate_buffer_size_in_frames_from_descriptor() (+358 more)

### Community 3 - "LLaMA Core/Adapters"
Cohesion: 0.01
Nodes (133): apply(), apply_to(), init(), llama_adapter_lora_init(), llama_adapter_lora_init_impl(), tensor_for(), llm_chat_apply_template(), llm_chat_detect_template() (+125 more)

### Community 4 - "Model Architectures"
Cohesion: 0.01
Nodes (39): altup_compute_router_modalities(), altup_correct(), altup_predict(), calc_magnitude(), gaussian_topk(), get_per_layer_inputs(), laurel(), llm_build_gemma3n_iswa() (+31 more)

### Community 5 - "LLaMA Batch Processing"
Cohesion: 0.01
Nodes (108): clear(), get_n_tokens(), init(), seq_pos_max(), seq_pos_min(), split_equal(), split_reset(), split_seq() (+100 more)

### Community 6 - "STB Image Loading"
Cohesion: 0.03
Nodes (194): load_jpeg_image(), stbi__addints_valid(), stbi__addsizes_valid(), stbi__at_eof(), stbi__bit_reverse(), stbi__bitcount(), stbi__bitreverse16(), stbi__blinn_8x8() (+186 more)

### Community 7 - "Cogito Parser"
Cohesion: 0.01
Nodes (94): cogitoEvent, cogitoEventContent, cogitoEventThinkingContent, cogitoEventToolCall, CogitoParser, CogitoParserState, CogitoRenderer, DeepSeek3Parser (+86 more)

### Community 8 - "Model Creation/Convert"
Cohesion: 0.02
Nodes (111): convertFromSafetensors(), convertModelFromFiles(), copyFile(), createConfigLayer(), createLink(), createModel(), detectModelTypeFromFiles(), ggufLayers() (+103 more)

### Community 9 - "Digest/GLM47 Tests"
Cohesion: 0.03
Nodes (121): formatGLM47ToolJSON(), GLM47Parser, GLM47Renderer, renderGLM47ToolArguments(), accept(), add(), array(), at() (+113 more)

### Community 10 - "Miniaudio Buffer Lifecycle"
Cohesion: 0.03
Nodes (152): ma_async_notification_signal(), ma_audio_buffer_config_init(), ma_audio_buffer_ref_uninit(), ma_audio_buffer_uninit(), ma_audio_buffer_uninit_and_free(), ma_audio_buffer_uninit_ex(), ma_copy_string(), ma_copy_string_w() (+144 more)

### Community 11 - "LLaMA Context/Decode"
Cohesion: 0.02
Nodes (118): apply_adapter_cvec(), attach_threadpool(), clear_adapter_lora(), decode(), detach_threadpool(), encode(), get_embeddings(), get_embeddings_ith() (+110 more)

### Community 12 - "LLaMA Vocabulary/Tokenizer"
Cohesion: 0.02
Nodes (97): byte_to_token(), detokenize(), get_add_bos(), get_add_eos(), get_add_sep(), get_pre_type(), get_type(), impl::detokenize() (+89 more)

### Community 13 - "CLIP Vision Encoder"
Cohesion: 0.02
Nodes (46): build_attn(), build_ffn(), build_inp(), build_inp_raw(), build_norm(), build_patch_merge_permute(), build_vit(), cb() (+38 more)

### Community 14 - "Miniaudio DSP/Volume"
Cohesion: 0.03
Nodes (135): ma_atomic_vec3f_get(), ma_atomic_vec3f_set(), ma_attenuation_inverse(), ma_attenuation_linear(), ma_audio_buffer_read_pcm_frames(), ma_audio_buffer_ref__data_source_on_read(), ma_audio_buffer_ref_read_pcm_frames(), ma_blend_f32() (+127 more)

### Community 15 - "LLaMA Sampling"
Cohesion: 0.02
Nodes (77): get_overlapping_token_sequences(), get_rng_seed(), llama_perf_sampler(), llama_perf_sampler_print(), llama_sample_dist(), llama_sample_xtc_apply(), llama_sampler_accept(), llama_sampler_apply() (+69 more)

### Community 16 - "Miniaudio WAV Codec"
Cohesion: 0.03
Nodes (127): ma_decoding_backend_uninit__wav(), ma_dr_wav_alaw_to_f32(), ma_dr_wav__alaw_to_s16(), ma_dr_wav_alaw_to_s32(), ma_dr_wav__bswap16(), ma_dr_wav__bswap_f32(), ma_dr_wav__bswap_s16(), ma_dr_wav__bswap_s32() (+119 more)

### Community 17 - "Common Utils/CLI"
Cohesion: 0.02
Nodes (50): common_context_params_to_llama(), common_control_vector_load(), common_control_vector_load_one(), common_init_from_params(), common_init_result(), common_init_sampler_from_model(), common_model_params_to_llama(), common_opt_get_optimizer() (+42 more)

### Community 18 - "Miniaudio PCM/SIMD"
Cohesion: 0.03
Nodes (118): ma_copy_and_apply_volume_factor_per_channel_f32(), ma_cpuid(), ma_dither_f32(), ma_dither_f32_rectangle(), ma_dither_f32_triangle(), ma_dither_s32(), ma_gainer_process_pcm_frames_internal(), ma_has_avx() (+110 more)

### Community 19 - "Miniaudio FLAC Decoder"
Cohesion: 0.05
Nodes (87): ma_dr_flac__calculate_prediction_32(), ma_dr_flac__calculate_prediction_64(), ma_dr_flac_crc8(), ma_dr_flac_crc8_byte(), ma_dr_flac__decode_flac_frame(), ma_dr_flac__decode_flac_frame_and_seek_forward_by_pcm_frames(), ma_dr_flac__decode_samples__constant(), ma_dr_flac__decode_samples__fixed() (+79 more)

### Community 20 - "Miniaudio WASAPI/Events"
Cohesion: 0.03
Nodes (86): ma_async_notification_event_init(), ma_async_notification_event_uninit(), ma_async_notification_event_wait(), ma_completion_handler_uwp_init(), ma_context_init_command__wasapi(), ma_context_post_command__wasapi(), ma_context_uninit__wasapi(), ma_decoder__on_tell_vfs() (+78 more)

### Community 21 - "Miniaudio VFS/File I/O"
Cohesion: 0.03
Nodes (81): ma_default_vfs_open(), ma_default_vfs_open__stdio(), ma_dr_flac_open_file_with_metadata(), ma_dr_flac_open_file_with_metadata_w(), ma_dr_mp3_init_file_w(), ma_dr_mp3_init_file_with_metadata_w(), ma_dr_wav_aiff_extented_to_s64(), ma_dr_wav_bytes_to_guid() (+73 more)

### Community 22 - "Miniaudio MP3 Decoder"
Cohesion: 0.04
Nodes (81): ma_decoding_backend_uninit__mp3(), ma_dr_mp3__accumulate_running_pcm_frame_count(), ma_dr_mp3_bs_get_bits(), ma_dr_mp3_bs_init(), ma_dr_mp3_calculate_seek_points(), ma_dr_mp3_copy_allocation_callbacks_or_defaults(), ma_dr_mp3_decode_next_frame(), ma_dr_mp3_decode_next_frame_ex() (+73 more)

### Community 23 - "Auth/Registry"
Cohesion: 0.05
Nodes (46): getAuthorizationToken(), registryChallenge, base64, base64_error, decode(), buildCloudSignatureChallenge(), cloudModelPathPassthroughMiddleware(), cloudPassthroughMiddleware() (+38 more)

### Community 24 - "Server/Scheduler"
Cohesion: 0.04
Nodes (32): assignLayers(), canRetry(), CompletionRequest, CompletionResponse, decodeUserJSON(), DoneReason, EmbeddingRequest, EmbeddingResponse (+24 more)

### Community 25 - "Miniaudio Atomic Ops"
Cohesion: 0.03
Nodes (72): ma_apply_volume_factor_f32(), ma_atomic_compare_and_swap_32(), ma_atomic_compare_and_swap_64(), ma_atomic_compare_and_swap_f32(), ma_atomic_compare_and_swap_f64(), ma_atomic_compare_and_swap_ptr(), ma_atomic_compare_exchange_strong_explicit_32(), ma_atomic_compare_exchange_strong_explicit_64() (+64 more)

### Community 26 - "Embedding Model"
Cohesion: 0.05
Nodes (28): Attention, embedModel, EncoderLayer, MLP, Model, Options, Batch, Input (+20 more)

### Community 27 - "Gemini Text/Altup"
Cohesion: 0.06
Nodes (26): AltUp, dense, Laurel, Layer, MLP, PerLayerProjector, SelfAttention, sparse (+18 more)

### Community 28 - "Cache Tests"
Cohesion: 0.06
Nodes (36): checkNotExists(), DiskCache, dumpCacheContents(), entryChecker(), errOnBangReader, mkdigest(), must(), openTester() (+28 more)

### Community 29 - "Vision Model Pipeline"
Cohesion: 0.06
Nodes (28): applyRotaryPositionEmbeddings(), applyVisionRotaryEmbedding(), blockDiagonalMask(), floorDiv(), Grid, PatchEmbedding, PatchMerger, PrecomputedAspectRatioEmbedding (+20 more)

### Community 30 - "Miniaudio Biquad Filter"
Cohesion: 0.05
Nodes (56): ma_biquad_node_process_pcm_frames(), ma_biquad_process_pcm_frame_f32(), ma_biquad_process_pcm_frame_f32__direct_form_2_transposed(), ma_biquad_process_pcm_frame_s16(), ma_biquad_process_pcm_frame_s16__direct_form_2_transposed(), ma_biquad_process_pcm_frames(), ma_bpf2_process_pcm_frame_f32(), ma_bpf2_process_pcm_frame_s16() (+48 more)

### Community 31 - "Miniaudio Decoder Init"
Cohesion: 0.08
Nodes (51): ma_decode_file(), ma_decode_from_vfs(), ma_decode_memory(), ma_decoder_config_init_copy(), ma_decoder_init(), ma_decoder_init_custom_from_file__internal(), ma_decoder_init_custom_from_file_w__internal(), ma_decoder_init_custom_from_memory__internal() (+43 more)

### Community 32 - "Miniaudio Volume Apply"
Cohesion: 0.05
Nodes (48): ma_apply_volume_unclipped_f32(), ma_apply_volume_unclipped_s16(), ma_apply_volume_unclipped_s24(), ma_apply_volume_unclipped_s32(), ma_apply_volume_unclipped_u8(), ma_clip_f32(), ma_clip_pcm_frames(), ma_clip_s16() (+40 more)

### Community 33 - "Miniaudio RNG/Dither"
Cohesion: 0.05
Nodes (48): ma_debug_fill_pcm_frames_with_sine_wave(), ma_lcg_rand_f32(), ma_lcg_rand_f64(), ma_lcg_rand_range_f32(), ma_lcg_rand_range_s32(), ma_lcg_rand_s16(), ma_lcg_rand_s32(), ma_lcg_rand_u32() (+40 more)

### Community 34 - "LLaMA Grammar"
Cohesion: 0.1
Nodes (38): add_rule(), c_rules(), decode_utf8(), generate_symbol_id(), get_symbol_id(), is_char_element(), is_digit_char(), is_eog() (+30 more)

### Community 35 - "Miniaudio FLAC Utils"
Cohesion: 0.06
Nodes (44): ma_decoding_backend_uninit__flac(), ma_dr_flac__be2host_16(), ma_dr_flac__be2host_32(), ma_dr_flac__be2host_64(), ma_dr_flac_close(), ma_dr_flac__cpuid(), ma_dr_flac_crc32_buffer(), ma_dr_flac_crc32_byte() (+36 more)

### Community 36 - "Scheduler Tests"
Cohesion: 0.05
Nodes (11): mockLlm, newScenarioRequest(), reqBundle, TestSchedAlreadyCanceled(), TestSchedExpireRunner(), TestSchedGetRunner(), TestSchedLoad(), TestSchedPrematureExpired() (+3 more)

### Community 37 - "LFM2 Architecture"
Cohesion: 0.08
Nodes (26): build_attn_block(), build_dense_feed_forward(), build_moe_feed_forward(), build_shortconv_block(), findMatchingParen(), isIdentPart(), isIdentStart(), lfm2Event (+18 more)

### Community 38 - "Image Processing"
Cohesion: 0.07
Nodes (15): ImageProcessor, findBestAspectRatio(), Grid, ImageProcessor, ProcessImage(), ratio, clipImageSize, cropImage() (+7 more)

### Community 39 - "Registry Tests"
Cohesion: 0.09
Nodes (33): checkErrCode(), checkRequest(), flushAfterWriter, importBytes(), newClient(), newRegistryClient(), recordRoundTripper, TestPullCached() (+25 more)

### Community 40 - "Miniaudio Channel Convert"
Cohesion: 0.1
Nodes (34): ma_calculate_channel_position_rectangular_weight(), ma_channel_converter_config_get_conversion_path(), ma_channel_converter_config_init(), ma_channel_converter_config_init_from_data_converter_config(), ma_channel_converter_float_to_fixed(), ma_channel_converter_get_heap_layout(), ma_channel_converter_get_heap_size(), ma_channel_converter_init() (+26 more)

### Community 41 - "Blob Cache"
Cohesion: 0.09
Nodes (11): absJoin(), checkWriter, DiskCache, Entry, HybridCache, nameToPath(), Open(), pathToName() (+3 more)

### Community 42 - "Miniaudio Engine Time"
Cohesion: 0.08
Nodes (32): ma_engine_get_time(), ma_engine_get_time_in_milliseconds(), ma_engine_get_time_in_pcm_frames(), ma_node_get_state_by_time(), ma_node_get_time(), ma_node_graph_get_time(), ma_node_set_state_time(), ma_sound_get_engine() (+24 more)

### Community 43 - "Miniaudio Ring Buffer"
Cohesion: 0.09
Nodes (30): ma_pcm_rb_acquire_write(), ma_pcm_rb_available_read(), ma_pcm_rb_available_write(), ma_pcm_rb_commit_write(), ma_pcm_rb_get_bpf(), ma_pcm_rb_get_subbuffer_offset(), ma_pcm_rb_get_subbuffer_ptr(), ma_pcm_rb_get_subbuffer_size() (+22 more)

### Community 44 - "FunctionGemma Parser"
Cohesion: 0.11
Nodes (7): functionGemmaEvent, FunctionGemmaEventContent, functionGemmaEventToolCall, FunctionGemmaParser, FunctionGemmaParserState, FunctionGemmaRenderer, parseNumber()

### Community 45 - "Scheduler Core"
Cohesion: 0.14
Nodes (5): ByDurationAndName, LlmRequest, runnerRef, Scheduler, schedulerModelKey()

### Community 46 - "Miniaudio MP3 SIMD"
Cohesion: 0.09
Nodes (27): ma_dr_mp3_clip_int16_arm(), ma_dr_mp3_have_simd(), ma_dr_mp3_L3_antialias(), ma_dr_mp3_L3_change_sign(), ma_dr_mp3_L3_dct3_9(), ma_dr_mp3_L3_decode(), ma_dr_mp3_L3_decode_scalefactors(), ma_dr_mp3_L3_huffman() (+19 more)

### Community 47 - "Miniaudio VFS Decoder"
Cohesion: 0.1
Nodes (25): ma_decoder__on_read_vfs(), ma_decoder__preinit_vfs(), ma_decoder__preinit_vfs_w(), ma_default_vfs_read(), ma_default_vfs_read__stdio(), ma_default_vfs_read__win32(), ma_encoder_init(), ma_encoder_init_file() (+17 more)

### Community 48 - "GLM46 Parser"
Cohesion: 0.11
Nodes (13): escapeGLM46Content(), glm46Event, glm46EventContent, glm46EventRawToolCall, glm46EventThinkingContent, GLM46Parser, glm46ParserState, GLM46Renderer (+5 more)

### Community 49 - "Miniaudio AAudio/Android"
Cohesion: 0.11
Nodes (24): ma_android_sdk_version(), ma_close_stream__aaudio(), ma_context_add_native_data_format_from_AAudioStream__aaudio(), ma_context_add_native_data_format_from_AAudioStream_ex__aaudio(), ma_context_enumerate_devices__aaudio(), ma_context_get_device_info__aaudio(), ma_create_and_configure_AAudioStreamBuilder__aaudio(), ma_device_get_info__aaudio() (+16 more)

### Community 50 - "Olmo3 Parser"
Cohesion: 0.13
Nodes (14): Olmo3Parser, olmo3ParserEvent, olmo3ParserEventContent, olmo3ParserEventToolCalls, olmo3ParserState, Olmo3Renderer, parseOlmo3Arguments(), parseOlmo3Array() (+6 more)

### Community 51 - "Miniaudio FLAC Stereo S16"
Cohesion: 0.09
Nodes (23): ma_dr_flac__mm_packs_interleaved_epi32(), ma_dr_flac_read_pcm_frames_s16__decode_independent_stereo(), ma_dr_flac_read_pcm_frames_s16__decode_independent_stereo__neon(), ma_dr_flac_read_pcm_frames_s16__decode_independent_stereo__reference(), ma_dr_flac_read_pcm_frames_s16__decode_independent_stereo__scalar(), ma_dr_flac_read_pcm_frames_s16__decode_independent_stereo__sse2(), ma_dr_flac_read_pcm_frames_s16__decode_left_side(), ma_dr_flac_read_pcm_frames_s16__decode_left_side__neon() (+15 more)

### Community 52 - "Download Manager"
Cohesion: 0.17
Nodes (6): blobDownload, blobDownloadPart, downloadBlob(), downloadOpts, jsonBlobDownloadPart, newBackoff()

### Community 53 - "Miniaudio FLAC Stereo F32"
Cohesion: 0.1
Nodes (21): ma_dr_flac_read_pcm_frames_f32__decode_independent_stereo(), ma_dr_flac_read_pcm_frames_f32__decode_independent_stereo__neon(), ma_dr_flac_read_pcm_frames_f32__decode_independent_stereo__reference(), ma_dr_flac_read_pcm_frames_f32__decode_independent_stereo__scalar(), ma_dr_flac_read_pcm_frames_f32__decode_independent_stereo__sse2(), ma_dr_flac_read_pcm_frames_f32__decode_left_side(), ma_dr_flac_read_pcm_frames_f32__decode_left_side__neon(), ma_dr_flac_read_pcm_frames_f32__decode_left_side__reference() (+13 more)

### Community 54 - "Model Backend Tests"
Cohesion: 0.1
Nodes (2): fakeBackend, fakeTensor

### Community 55 - "Qwen35 Parser Tests"
Cohesion: 0.1
Nodes (4): qwen35MathTools(), qwen35WeatherUVTools(), TestQwen35RendererBackToBackToolCallsAndResponses(), TestQwen35RendererInterleavedThinkingAndTools()

### Community 56 - "Miniaudio Volume Frames"
Cohesion: 0.1
Nodes (20): ma_apply_volume_factor_pcm_frames(), ma_apply_volume_factor_pcm_frames_f32(), ma_apply_volume_factor_pcm_frames_s16(), ma_apply_volume_factor_pcm_frames_s24(), ma_apply_volume_factor_pcm_frames_s32(), ma_apply_volume_factor_pcm_frames_u8(), ma_apply_volume_factor_s16(), ma_apply_volume_factor_s24() (+12 more)

### Community 57 - "Routes Create Tests"
Cohesion: 0.27
Nodes (18): checkFileExists(), createBinFile(), createRequest(), NewRecorder(), responseRecorder, TestCreateAndShowRemoteModel(), TestCreateDetectTemplate(), TestCreateFromBin() (+10 more)

### Community 58 - "SAM Vision Model"
Cohesion: 0.16
Nodes (9): LayerNorm2D, relativeCoordinates(), relativePositions(), samAttention, samBlock, samMLP, samModel, samNeck (+1 more)

### Community 59 - "Line Reader"
Cohesion: 0.16
Nodes (5): Line, NewRelayReader(), RelayReader, relayWriter, Ticket

### Community 60 - "Miniaudio FLAC Open"
Cohesion: 0.13
Nodes (18): ma_dr_flac_open(), ma_dr_flac_open_and_read_pcm_frames_f32(), ma_dr_flac_open_and_read_pcm_frames_s16(), ma_dr_flac_open_and_read_pcm_frames_s32(), ma_dr_flac_open_file(), ma_dr_flac_open_file_and_read_pcm_frames_f32(), ma_dr_flac_open_file_and_read_pcm_frames_s16(), ma_dr_flac_open_file_and_read_pcm_frames_s32() (+10 more)

### Community 61 - "Miniaudio FLAC CRC"
Cohesion: 0.16
Nodes (18): ma_dr_flac__clz(), ma_dr_flac__clz_msvc(), ma_dr_flac__clz_software(), ma_dr_flac_crc16(), ma_dr_flac_crc16__32bit(), ma_dr_flac_crc16__64bit(), ma_dr_flac_crc16_byte(), ma_dr_flac_crc16_bytes() (+10 more)

### Community 62 - "Cloud Proxy Tests"
Cohesion: 0.14
Nodes (5): chunkRecorder, TestCloudPassthroughMiddleware_ZstdBody(), TestCloudPassthroughMiddleware_ZstdBodyTooLarge(), TestJSONLFramingResponseWriter_FlushPendingWritesTrailingLine(), TestJSONLFramingResponseWriter_SplitsCoalescedLines()

### Community 63 - "Image Processing Tests"
Cohesion: 0.12
Nodes (0): 

### Community 64 - "Qwen3Coder Tests"
Cohesion: 0.13
Nodes (2): TestQwenToolParser(), tool()

### Community 65 - "LFM2 Tests"
Cohesion: 0.13
Nodes (0): 

### Community 66 - "Qwen3VL Thinking Tests"
Cohesion: 0.14
Nodes (0): 

### Community 67 - "Qwen3 Tests"
Cohesion: 0.15
Nodes (0): 

### Community 68 - "README/Vendoring Docs"
Cohesion: 0.18
Nodes (12): FETCH_HEAD Base Commit Pin, ggml, llama.cpp, llama Go Bindings Package, Makefile.sync, Patch Workflow, vendor/ Directory, Vendoring Rationale (+4 more)

### Community 69 - "Miniaudio FLAC Stereo S32"
Cohesion: 0.18
Nodes (11): ma_dr_flac_read_pcm_frames_s32__decode_independent_stereo(), ma_dr_flac_read_pcm_frames_s32__decode_independent_stereo__neon(), ma_dr_flac_read_pcm_frames_s32__decode_independent_stereo__reference(), ma_dr_flac_read_pcm_frames_s32__decode_independent_stereo__scalar(), ma_dr_flac_read_pcm_frames_s32__decode_independent_stereo__sse2(), ma_dr_flac_read_pcm_frames_s32__decode_mid_side(), ma_dr_flac_read_pcm_frames_s32__decode_mid_side__neon(), ma_dr_flac_read_pcm_frames_s32__decode_mid_side__reference() (+3 more)

### Community 70 - "Miniaudio FLAC Left Side"
Cohesion: 0.18
Nodes (11): ma_dr_flac_read_pcm_frames_s32__decode_left_side(), ma_dr_flac_read_pcm_frames_s32__decode_left_side__neon(), ma_dr_flac_read_pcm_frames_s32__decode_left_side__reference(), ma_dr_flac_read_pcm_frames_s32__decode_left_side__scalar(), ma_dr_flac_read_pcm_frames_s32__decode_left_side__sse2(), ma_dr_flac_read_pcm_frames_s32__decode_right_side(), ma_dr_flac_read_pcm_frames_s32__decode_right_side__neon(), ma_dr_flac_read_pcm_frames_s32__decode_right_side__reference() (+3 more)

### Community 71 - "Miniaudio WAV Byte Swap"
Cohesion: 0.2
Nodes (11): ma_dr_wav__bswap64(), ma_dr_wav__bswap_s24(), ma_dr_wav__bswap_s64(), ma_dr_wav__bswap_samples(), ma_dr_wav__bswap_samples_s24(), ma_dr_wav__bswap_samples_s64(), ma_dr_wav_write_pcm_frames(), ma_dr_wav_write_pcm_frames_be() (+3 more)

### Community 72 - "Olmo3 Think Tests"
Cohesion: 0.18
Nodes (0): 

### Community 73 - "Parser Framework Tests"
Cohesion: 0.22
Nodes (3): mockParser, TestOverrideBuiltInParser(), TestRegisterCustomParser()

### Community 74 - "Digest Utils"
Cohesion: 0.25
Nodes (2): Digest, ParseDigest()

### Community 75 - "Miniaudio Resampler Rate"
Cohesion: 0.2
Nodes (10): ma_gcf_u32(), ma_linear_resampler_adjust_timer_for_new_rate(), ma_linear_resampler_set_rate(), ma_linear_resampler_set_rate_internal(), ma_linear_resampler_set_rate_ratio(), ma_lpf_config_init(), ma_lpf_node_config_init(), ma_lpf_node_reinit() (+2 more)

### Community 76 - "Miniaudio PCM Interleave"
Cohesion: 0.22
Nodes (9): ma_copy_memory_64(), ma_pcm_f32_to_f32(), ma_pcm_interleave_u8(), ma_pcm_interleave_u8__optimized(), ma_pcm_interleave_u8__reference(), ma_pcm_s16_to_s16(), ma_pcm_s24_to_s24(), ma_pcm_s32_to_s32() (+1 more)

### Community 77 - "Miniaudio Resource Manager"
Cohesion: 0.22
Nodes (9): ma_resource_manager_data_buffer_init(), ma_resource_manager_data_buffer_init_copy(), ma_resource_manager_data_buffer_init_ex(), ma_resource_manager_data_buffer_init_w(), ma_resource_manager_data_source_init(), ma_resource_manager_data_source_init_copy(), ma_resource_manager_data_source_init_ex(), ma_resource_manager_data_source_init_w() (+1 more)

### Community 78 - "DeepSeek3 Tests"
Cohesion: 0.22
Nodes (0): 

### Community 79 - "GLM47 Tests"
Cohesion: 0.22
Nodes (0): 

### Community 80 - "Nemotron3Nano Tests"
Cohesion: 0.22
Nodes (0): 

### Community 81 - "Inference Request Logger"
Cohesion: 0.31
Nodes (4): inferenceRequestLogger, newInferenceRequestLogger(), sanitizeRouteForFilename(), Server

### Community 82 - "Quantization"
Cohesion: 0.36
Nodes (7): getTensorNewType(), newType(), quantize(), quantizer, quantizeState, qwen3LinearAttnQuantType(), useMoreBits()

### Community 83 - "Miniaudio Volume Set"
Cohesion: 0.25
Nodes (8): ma_device_set_master_volume(), ma_device_set_master_volume_db(), ma_engine_set_gain_db(), ma_engine_set_volume(), ma_node_output_bus_set_volume(), ma_node_set_output_bus_volume(), ma_powf(), ma_volume_db_to_linear()

### Community 84 - "Miniaudio Volume Get"
Cohesion: 0.25
Nodes (8): ma_device_get_master_volume(), ma_device_get_master_volume_db(), ma_engine_get_gain_db(), ma_engine_get_volume(), ma_log10d(), ma_log10f(), ma_logd(), ma_volume_linear_to_db()

### Community 85 - "DeltaNet Architecture"
Cohesion: 0.36
Nodes (4): convKernel, createMasks(), GatedDeltaNet, Masks

### Community 86 - "Cogito Tests"
Cohesion: 0.25
Nodes (0): 

### Community 87 - "Olmo3 Tests"
Cohesion: 0.25
Nodes (0): 

### Community 88 - "Renderer Framework"
Cohesion: 0.29
Nodes (5): Renderer, RendererConstructor, rendererForName(), RendererRegistry, RenderWithRenderer()

### Community 89 - "Quantization Tests"
Cohesion: 0.36
Nodes (4): cosineSimilarity(), dotProduct(), magnitude(), TestConvertToF32()

### Community 90 - "Chunked Transfer"
Cohesion: 0.29
Nodes (3): Chunk, Chunker, DiskCache

### Community 91 - "Name Parser Tests"
Cohesion: 0.25
Nodes (0): 

### Community 92 - "Miniaudio Resource Register"
Cohesion: 0.29
Nodes (7): ma_resource_manager_register_data(), ma_resource_manager_register_decoded_data(), ma_resource_manager_register_decoded_data_internal(), ma_resource_manager_register_decoded_data_w(), ma_resource_manager_register_encoded_data(), ma_resource_manager_register_encoded_data_internal(), ma_resource_manager_register_encoded_data_w()

### Community 93 - "Renderer Tests"
Cohesion: 0.29
Nodes (1): mockRenderer

### Community 94 - "Miniaudio Config Init"
Cohesion: 0.33
Nodes (6): ma_decoder_config_init(), ma_resampler_config_init(), ma_sound_group_config_init_2(), ma_sound_group_init(), ma_sound_group_init_ex(), void()

### Community 95 - "Miniaudio Noise Gen"
Cohesion: 0.4
Nodes (6): ma_lcg_seed(), ma_noise_get_heap_layout(), ma_noise_get_heap_size(), ma_noise_init(), ma_noise_init_preallocated(), ma_seed()

### Community 96 - "Miniaudio Filter Reset"
Cohesion: 0.33
Nodes (6): ma_biquad_clear_cache(), ma_linear_resampler_reset(), ma_lpf1_clear_cache(), ma_lpf2_clear_cache(), ma_lpf_clear_cache(), ma_resampling_backend_reset__linear()

### Community 97 - "Miniaudio Loop Control"
Cohesion: 0.33
Nodes (6): ma_data_source_is_looping(), ma_data_source_node_is_looping(), ma_resource_manager_data_buffer_is_looping(), ma_resource_manager_data_source_is_looping(), ma_resource_manager_data_stream_is_looping(), ma_sound_is_looping()

### Community 98 - "Model Config Tests"
Cohesion: 0.33
Nodes (0): 

### Community 99 - "GLM46 Tests"
Cohesion: 0.33
Nodes (0): 

### Community 100 - "Ministral Tests"
Cohesion: 0.33
Nodes (0): 

### Community 101 - "Qwen3VL NonThink Tests"
Cohesion: 0.33
Nodes (0): 

### Community 102 - "Tracing/Telemetry"
Cohesion: 0.47
Nodes (4): Trace, traceFromContext(), traceKey, WithTrace()

### Community 103 - "Miniaudio Sound Cursor"
Cohesion: 0.4
Nodes (5): ma_sound_get_cursor_in_pcm_frames(), ma_sound_get_cursor_in_seconds(), ma_sound_get_data_format(), ma_sound_seek_to_pcm_frame(), ma_sound_seek_to_second()

### Community 104 - "Miniaudio COM/GUID"
Cohesion: 0.4
Nodes (5): ma_completion_handler_uwp_QueryInterface(), ma_context_get_device_info_callback__dsound(), ma_IMMNotificationClient_QueryInterface(), ma_is_guid_equal(), ma_is_guid_null()

### Community 105 - "Miniaudio Atomic Fetch"
Cohesion: 0.4
Nodes (5): ma_atomic_fetch_add_explicit_32(), ma_atomic_fetch_add_explicit_f32(), ma_atomic_fetch_sub_explicit_32(), ma_atomic_fetch_sub_explicit_f32(), ma_atomic_thread_fence()

### Community 106 - "Miniaudio Latency Query"
Cohesion: 0.4
Nodes (5): ma_linear_resampler_get_input_latency(), ma_linear_resampler_get_output_latency(), ma_lpf_get_latency(), ma_resampling_backend_get_input_latency__linear(), ma_resampling_backend_get_output_latency__linear()

### Community 107 - "Miniaudio Atomic Store"
Cohesion: 0.4
Nodes (5): ma_atomic_store_explicit_32(), ma_atomic_store_explicit_64(), ma_atomic_store_explicit_f32(), ma_atomic_store_explicit_f64(), ma_atomic_store_explicit_ptr()

### Community 108 - "Attention Mechanism"
Cohesion: 0.4
Nodes (2): Attention, FullAttention

### Community 109 - "Auth Tests"
Cohesion: 0.4
Nodes (0): 

### Community 110 - "Sync Group Utils"
Cohesion: 0.4
Nodes (1): Group

### Community 111 - "Status Writer"
Cohesion: 0.5
Nodes (1): StatusWriter

### Community 112 - "Short Convolution"
Cohesion: 0.5
Nodes (2): ShortConv, shortConvKernel

### Community 113 - "Mamba2 SSM"
Cohesion: 0.5
Nodes (2): convKernel, Mamba2

### Community 114 - "Model Validate Tests"
Cohesion: 0.5
Nodes (0): 

### Community 115 - "FunctionGemma Tests"
Cohesion: 0.5
Nodes (0): 

### Community 116 - "Prompt Rendering"
Cohesion: 0.67
Nodes (3): chatPrompt(), renderPrompt(), tokenizeFunc

### Community 117 - "Request Log Tests"
Cohesion: 0.5
Nodes (0): 

### Community 118 - "Case Check Tests"
Cohesion: 0.83
Nodes (3): isCaseSensitive(), isCI(), useCaseInsensitiveTempDir()

### Community 119 - "LLaMA Binding Tests"
Cohesion: 0.67
Nodes (0): 

### Community 120 - "Miniaudio Interleave S16"
Cohesion: 1.0
Nodes (3): ma_pcm_interleave_s16(), ma_pcm_interleave_s16__optimized(), ma_pcm_interleave_s16__reference()

### Community 121 - "Miniaudio Interleave S32"
Cohesion: 1.0
Nodes (3): ma_pcm_interleave_s32(), ma_pcm_interleave_s32__optimized(), ma_pcm_interleave_s32__reference()

### Community 122 - "Miniaudio Deinterleave S32"
Cohesion: 1.0
Nodes (3): ma_pcm_deinterleave_s32(), ma_pcm_deinterleave_s32__optimized(), ma_pcm_deinterleave_s32__reference()

### Community 123 - "Miniaudio Deinterleave F32"
Cohesion: 1.0
Nodes (3): ma_pcm_deinterleave_f32(), ma_pcm_deinterleave_f32__optimized(), ma_pcm_deinterleave_f32__reference()

### Community 124 - "Miniaudio Deinterleave S24"
Cohesion: 1.0
Nodes (3): ma_pcm_deinterleave_s24(), ma_pcm_deinterleave_s24__optimized(), ma_pcm_deinterleave_s24__reference()

### Community 125 - "FixBlobs Tests"
Cohesion: 1.0
Nodes (2): slurpFiles(), TestFixBlobs()

### Community 126 - "LogProb Utils"
Cohesion: 1.0
Nodes (2): stringToByteInts(), toAPILogprobs()

### Community 127 - "Model Resolver Tests"
Cohesion: 0.67
Nodes (0): 

### Community 128 - "Backoff Tests"
Cohesion: 0.67
Nodes (0): 

### Community 129 - "GLM OCR Tests"
Cohesion: 1.0
Nodes (0): 

### Community 130 - "JSON Tests"
Cohesion: 1.0
Nodes (0): 

### Community 131 - "FixBlobs"
Cohesion: 1.0
Nodes (0): 

### Community 132 - "Routes Options Tests"
Cohesion: 1.0
Nodes (0): 

### Community 133 - "Sparse Common"
Cohesion: 1.0
Nodes (0): 

### Community 134 - "Sparse Windows"
Cohesion: 1.0
Nodes (0): 

### Community 135 - "Test Home Helper"
Cohesion: 1.0
Nodes (0): 

### Community 136 - "Registry Synctest"
Cohesion: 1.0
Nodes (0): 

### Community 137 - "Backoff Synctest"
Cohesion: 1.0
Nodes (0): 

### Community 138 - "StringsX Tests"
Cohesion: 1.0
Nodes (0): 

### Community 139 - "Line Reader Tests"
Cohesion: 1.0
Nodes (0): 

### Community 140 - "Build Info"
Cohesion: 1.0
Nodes (0): 

### Community 141 - "LLM Darwin"
Cohesion: 1.0
Nodes (0): 

### Community 142 - "LLM Linux"
Cohesion: 1.0
Nodes (0): 

### Community 143 - "LLM Windows"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **112 isolated node(s):** `Devices`, `ContextParams`, `ModelParams`, `MtmdChunk`, `SamplingParams` (+107 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `GLM OCR Tests`** (2 nodes): `glmocr_test.go`, `TestGlmOcrRenderer_Images()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `JSON Tests`** (2 nodes): `json_test.go`, `TestMarshalWithSpaces()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FixBlobs`** (2 nodes): `fixblobs.go`, `fixBlobs()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Routes Options Tests`** (2 nodes): `routes_options_test.go`, `TestModelOptionsNumCtxPriority()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sparse Common`** (2 nodes): `sparse_common.go`, `setSparse()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sparse Windows`** (2 nodes): `sparse_windows.go`, `setSparse()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Home Helper`** (2 nodes): `test_home_test.go`, `setTestHome()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Registry Synctest`** (2 nodes): `registry_synctest_test.go`, `TestPullDownloadTimeout()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backoff Synctest`** (2 nodes): `backoff_synctest_test.go`, `TestLoop()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `StringsX Tests`** (2 nodes): `stringsx_test.go`, `TestCompareFold()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Line Reader Tests`** (2 nodes): `line_test.go`, `TestPipelineReadWriterTo()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Build Info`** (1 nodes): `build-info.cpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LLM Darwin`** (1 nodes): `llm_darwin.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LLM Linux`** (1 nodes): `llm_linux.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LLM Windows`** (1 nodes): `llm_windows.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Are the 143 inferred relationships involving `ma_free()` (e.g. with `ma_wfopen()` and `ma__free_default()`) actually correct?**
  _`ma_free()` has 143 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `ma_malloc()` (e.g. with `ma_copy_string()` and `ma_copy_string_w()`) actually correct?**
  _`ma_malloc()` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `ma_get_bytes_per_frame()` (e.g. with `ma_get_bytes_per_sample()` and `ma_device__handle_data_callback()`) actually correct?**
  _`ma_get_bytes_per_frame()` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `ma_log_postf()` (e.g. with `ma_log_postv()` and `ma_dlopen()`) actually correct?**
  _`ma_log_postf()` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `ma_device_get_log()` (e.g. with `ma_device__handle_duplex_callback_capture()` and `ma_IMMNotificationClient_OnDefaultDeviceChanged()`) actually correct?**
  _`ma_device_get_log()` has 53 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Devices`, `ContextParams`, `ModelParams` to the rest of the system?**
  _112 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Miniaudio Async/Events` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._