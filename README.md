# HMM-rust

This is an implementation of Hidden MArkov Model in rust.

The HMM instance provides four algorithms - Viterbi, Forward, Backward and Baum-Welch training.
The probabilities of the model are considered to be from discrete distribution.
The model does take for granted that all the probabilities are non-zero.

The tests for every implemented function are also provided. They also show the basic usage of the HMM struct.
