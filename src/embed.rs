//! The embedding seam: the engine knows a model only by `model_id`, `dimension` and
//! its two retrieval prompts (SPEC §6). Vectors may come back unnormalised; the engine
//! normalises them and rejects any whose length contradicts `dimension`.

use crate::Result;

/// Text → vectors under one model.
pub trait Embedder: Send {
    /// Identifies the vector space; changing the model (or its dimension) must change
    /// it, so stored vectors are re-embedded instead of mixed.
    fn model_id(&self) -> &str;
    /// Vector length.
    fn dimension(&self) -> usize;
    /// Embeds texts to be stored (the model's document prompt).
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embeds search cues (the model's query prompt).
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

impl<E: Embedder + ?Sized> Embedder for Box<E> {
    fn model_id(&self) -> &str {
        (**self).model_id()
    }
    fn dimension(&self) -> usize {
        (**self).dimension()
    }
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_documents(texts)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed_queries(texts)
    }
}

/// Prepends a retrieval prompt to every text.
#[cfg(any(feature = "http", feature = "llama"))]
fn prefixed(prefix: &str, texts: &[&str]) -> Vec<String> {
    texts.iter().map(|t| format!("{prefix}{t}")).collect()
}

#[cfg(feature = "http")]
pub mod http {
    //! Any OpenAI-compatible `/embeddings` endpoint (llama-server, Ollama, LM Studio, OpenAI…).

    use super::{Embedder, prefixed};
    use crate::Result;
    use serde_json::{Value, json};

    /// Embeds through `POST {base_url}/embeddings`. Clones share the connection pool.
    #[derive(Clone)]
    pub struct HttpEmbedder {
        agent: ureq::Agent,
        url: String,
        model: String,
        api_key: String,
        prefixes: (String, String),
        dimension: usize,
    }

    impl HttpEmbedder {
        /// Connects and learns the dimension from one probe embedding.
        pub fn connect(base_url: &str, model: &str, api_key: &str, query_prefix: &str, document_prefix: &str) -> Result<Self> {
            let mut me = Self {
                agent: ureq::Agent::config_builder()
                    .timeout_global(Some(std::time::Duration::from_secs(120)))
                    .build()
                    .into(),
                url: format!("{}/embeddings", base_url.trim_end_matches('/')),
                model: model.to_owned(),
                api_key: api_key.to_owned(),
                prefixes: (query_prefix.to_owned(), document_prefix.to_owned()),
                dimension: 0,
            };
            me.dimension = me.post(vec!["dimension probe".to_owned()])?.first().map_or(0, Vec::len);
            if me.dimension == 0 {
                return Err("embedding endpoint returned an empty vector".into());
            }
            Ok(me)
        }

        fn post(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
            let mut req = self.agent.post(&self.url).header("Content-Type", "application/json");
            if !self.api_key.is_empty() {
                req = req.header("Authorization", &format!("Bearer {}", self.api_key));
            }
            let body: Value = serde_json::from_str(
                &req.send(json!({"model": self.model, "input": input}).to_string())?.body_mut().read_to_string()?,
            )?;
            let mut data: Vec<&Value> = body["data"].as_array().ok_or("no `data` in embedding response")?.iter().collect();
            data.sort_by_key(|d| d["index"].as_u64().unwrap_or(0));
            data.iter()
                .map(|d| {
                    d["embedding"].as_array().ok_or_else(|| "no `embedding`".into()).map(|v| v.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                })
                .collect()
        }
    }

    impl Embedder for HttpEmbedder {
        fn model_id(&self) -> &str {
            &self.model
        }
        fn dimension(&self) -> usize {
            self.dimension
        }
        fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.post(prefixed(&self.prefixes.1, texts))
        }
        fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.post(prefixed(&self.prefixes.0, texts))
        }
    }
}

#[cfg(feature = "llama")]
pub mod llama {
    //! Any GGUF embedding model run in-process by llama.cpp, fully offline.

    use super::{Embedder, prefixed};
    use crate::Result;
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::model::params::LlamaModelParams;
    use std::num::NonZeroU32;
    use std::path::Path;
    use std::sync::OnceLock;
    use std::sync::mpsc::{Sender, channel};

    /// Tokens per sequence (longer inputs are truncated) and per decode.
    const N_BATCH: u32 = 512;
    /// Sequences decoded together.
    const N_SEQ: u32 = 8;

    type Job = (Vec<String>, Sender<std::result::Result<Vec<Vec<f32>>, String>>);

    fn backend() -> std::result::Result<&'static LlamaBackend, String> {
        static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                LlamaBackend::init().map(|mut b| {
                    b.void_logs();
                    b
                }).map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// A GGUF embedding model. llama.cpp contexts are not `Send`, so the model and its
    /// context live on a worker thread that this handle talks to; clones share the worker.
    #[derive(Clone)]
    pub struct LlamaEmbedder {
        model_id: String,
        dimension: usize,
        prefixes: (String, String),
        jobs: Sender<Job>,
    }

    impl LlamaEmbedder {
        /// Loads `path` (the model's own training context; CPU threads = cores).
        pub fn load(path: &Path, query_prefix: &str, document_prefix: &str) -> Result<Self> {
            let backend = backend()?;
            let (jobs, inbox) = channel::<Job>();
            let (ready_tx, ready) = channel::<std::result::Result<usize, String>>();
            let model_id = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            let path = path.to_path_buf();
            std::thread::Builder::new().name("llama-embed".into()).spawn(move || {
                let setup = || -> std::result::Result<_, String> {
                    if !path.exists() {
                        return Err(format!("embedding model not found: {}", path.display()));
                    }
                    // With CUDA every layer goes to the GPU; the CPU build keeps them on the CPU.
                    let params = LlamaModelParams::default().with_n_gpu_layers(if cfg!(feature = "cuda") { u32::MAX } else { 0 });
                    let model = LlamaModel::load_from_file(backend, &path, &params).map_err(|e| e.to_string())?;
                    Ok(model)
                };
                let model = match setup() {
                    Ok(m) => m,
                    Err(e) => return drop(ready_tx.send(Err(e))),
                };
                let threads = crate::cpu_budget() as i32;
                let params = LlamaContextParams::default()
                    .with_embeddings(true)
                    .with_n_ctx(NonZeroU32::new(model.n_ctx_train().max(N_BATCH)))
                    .with_n_batch(N_BATCH)
                    .with_n_ubatch(N_BATCH)
                    .with_n_seq_max(N_SEQ)
                    .with_n_threads(threads)
                    .with_n_threads_batch(threads);
                let mut ctx = match model.new_context(backend, params) {
                    Ok(c) => c,
                    Err(e) => return drop(ready_tx.send(Err(e.to_string()))),
                };
                let dim = model.n_embd_out() as usize;
                if ready_tx.send(Ok(dim)).is_err() {
                    return;
                }
                let mut batch = LlamaBatch::new(N_BATCH as usize, N_SEQ as i32);
                for (texts, reply) in inbox {
                    let mut out = Vec::with_capacity(texts.len());
                    let mut run = |batch: &mut LlamaBatch, n: usize, out: &mut Vec<Vec<f32>>| -> std::result::Result<(), String> {
                        ctx.clear_kv_cache();
                        ctx.decode(batch).map_err(|e| e.to_string())?;
                        for s in 0..n {
                            out.push(ctx.embeddings_seq_ith(s as i32).map_err(|e| e.to_string())?.to_vec());
                        }
                        batch.clear();
                        Ok(())
                    };
                    let result = (|| {
                        let (mut seqs, mut toks) = (0usize, 0usize);
                        for t in &texts {
                            let mut tokens = model.vocab().tokenize(t.as_bytes(), true, false);
                            tokens.truncate(N_BATCH as usize);
                            if toks + tokens.len() > N_BATCH as usize || seqs == N_SEQ as usize {
                                run(&mut batch, seqs, &mut out)?;
                                (seqs, toks) = (0, 0);
                            }
                            batch.add_sequence(&tokens, seqs as i32, true).map_err(|e| e.to_string())?;
                            seqs += 1;
                            toks += tokens.len();
                        }
                        if seqs > 0 {
                            run(&mut batch, seqs, &mut out)?;
                        }
                        Ok(out)
                    })();
                    if result.is_err() {
                        batch.clear();
                    }
                    let _ = reply.send(result);
                }
            })?;
            let dimension = ready.recv().map_err(|_| "embedding worker died")??;
            Ok(Self { model_id, dimension, prefixes: (query_prefix.to_owned(), document_prefix.to_owned()), jobs })
        }

        fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            let (tx, rx) = channel();
            self.jobs.send((texts, tx)).map_err(|_| "embedding worker stopped")?;
            Ok(rx.recv().map_err(|_| "embedding worker stopped")??)
        }
    }

    impl Embedder for LlamaEmbedder {
        fn model_id(&self) -> &str {
            &self.model_id
        }
        fn dimension(&self) -> usize {
            self.dimension
        }
        fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.embed(prefixed(&self.prefixes.1, texts))
        }
        fn embed_queries(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.embed(prefixed(&self.prefixes.0, texts))
        }
    }
}
