from tokenizers import Tokenizer

test_tokenizer = Tokenizer.from_file('s1-tokenizer-32k.json')
prompt = "hi anubhav"

encoded = test_tokenizer.encode(prompt)

print(f"Original prompt: {prompt}")
print(encoded.tokens)

decoded_text = test_tokenizer.decode(encoded.ids)
print(decoded_text)
