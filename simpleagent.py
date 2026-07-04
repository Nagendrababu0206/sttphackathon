def search(query):
    return f"Searching for: {query}"

question = input("Question: ")

if "weather" in question.lower():
    print(search(question))
else:
    print("General Answer")
    