


1. norm mul 融合
cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
cb(cur, "attn_norm", il);


1. 矩阵融合，不考虑lora

// Input projections
auto qkvz = build_qkvz(cur, il);

ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);


ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    

build_layer_attn_linear