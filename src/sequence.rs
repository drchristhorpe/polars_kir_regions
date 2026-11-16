//! Sequence cleaning, transcribed from `histo_kir_hmm`'s `_clean_sequence`.
//!
//! This is the one utility `chunk` and `loops` needed from `polars_kir_hmm`'s classifier module,
//! and it is 20 lines of character filtering with no model or HMM in it. Copying it here is what
//! makes this crate standalone; depending on that package for it would have reintroduced the
//! coupling this split exists to remove.
//!
//! Cleaning matters for a region slice, not just for scoring: every boundary in the vendored
//! table indexes the **cleaned** sequence, so a stray gap character or a lowercase residue in the
//! input would shift every position after it if it were not removed first. The rule is the
//! reference's: uppercase, then keep only the 20 amino acids and `X`.

/// `c in AA_TO_IDX or c == "X"` -- the 20 amino acids plus the unknown-residue placeholder.
#[inline]
fn is_kept(byte: u8) -> bool {
    matches!(
        byte,
        b'A' | b'C'
            | b'D'
            | b'E'
            | b'F'
            | b'G'
            | b'H'
            | b'I'
            | b'K'
            | b'L'
            | b'M'
            | b'N'
            | b'P'
            | b'Q'
            | b'R'
            | b'S'
            | b'T'
            | b'V'
            | b'W'
            | b'Y'
            | b'X'
    )
}

/// Uppercase, then drop everything that is not an amino acid or `X`.
///
/// The non-ASCII path is deliberate and matches the reference: it uppercases with Python's
/// `str.upper()`, which is Unicode-aware and can map one character to several. An ASCII-only
/// uppercase would drop those instead and silently return a shorter sequence -- which for a
/// region slice means every boundary after the character is off by one, with no error raised.
pub fn clean_sequence(sequence: &str) -> Vec<u8> {
    if sequence.is_ascii() {
        return sequence
            .bytes()
            .map(|byte| byte.to_ascii_uppercase())
            .filter(|byte| is_kept(*byte))
            .collect();
    }
    sequence
        .to_uppercase()
        .bytes()
        .filter(|byte| is_kept(*byte))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::clean_sequence;

    #[test]
    fn uppercases_and_drops_non_residues() {
        assert_eq!(clean_sequence("hi s-e q*"), b"HISEQ".to_vec());
    }

    #[test]
    fn keeps_x_but_not_other_letters() {
        // B, J, O, U and Z are letters that are not amino acids; X is the unknown placeholder.
        assert_eq!(clean_sequence("AXBJOUZ"), b"AX".to_vec());
    }

    #[test]
    fn non_ascii_takes_the_unicode_path() {
        // 'ß'.upper() is "SS" in Python, so the reference keeps two residues here. An ASCII-only
        // uppercase would drop the character and shift every downstream boundary by two.
        assert_eq!(clean_sequence("AßA"), b"ASSA".to_vec());
    }
}
