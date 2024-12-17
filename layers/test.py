import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf


test_sentence = 'I went Shopping Today, and my trolly was filled with Bananas. I also had food at burgur palace'

aug = nac.KeyboardAug(name='Keyboard_Aug', aug_char_min=1, aug_char_max=10, aug_char_p=0.3, aug_word_p=0.3,
                      aug_word_min=1, aug_word_max=10, tokenizer=None, reverse_tokenizer=None,
                      include_special_char=True, include_numeric=True, include_upper_case=True, lang='en', verbose=0,
                      min_char=4)

test_sentence_aug = aug.augment(test_sentence)
print(test_sentence)
print(test_sentence_aug)

aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10, aug_p=0.3,
                     lang='eng',
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False,
                     verbose=0)

test_sentence_aug = aug.augment(test_sentence)
print(test_sentence)
print(test_sentence_aug)