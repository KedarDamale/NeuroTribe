"""TRIBE v2 integration.

TRIBE v2 is used **as a pretrained normative encoding model**. It is never
fine-tuned, retrained, or re-implemented here: this package wraps the official
`facebookresearch/tribev2` inference API and pins the commit/model revision it
used into every provenance manifest.

Licence note: the official TRIBE v2 repository is CC-BY-NC-4.0, so V1 of this
system is research / non-commercial.
"""
