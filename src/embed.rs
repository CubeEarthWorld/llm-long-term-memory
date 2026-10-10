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
