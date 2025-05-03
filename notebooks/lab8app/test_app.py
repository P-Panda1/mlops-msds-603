# test_app.py
import requests

data = {
    "text": "In a realm where celluloid dreams intertwine with the echoes of reincarnated destinies, Om Shanti Om emerges as a cinematic tapestry that defies conventional storytelling. The film navigates the labyrinthine corridors of Bollywood's golden era, weaving a narrative that is as much a homage as it is a parody. The protagonist's journey, marked by aspirations and untimely demise, sets the stage for a reincarnation saga that challenges the boundaries of time and identity. This duality—of life and afterlife, of past and present—serves as a metaphor for the ever-evolving nature of cinema itself. The film's aesthetic choices, from its vibrant costumes to its elaborate set designs, evoke a sense of nostalgia while simultaneously embracing modernity. The musical sequences, reminiscent of classic Bollywood numbers, are choreographed with a flair that pays tribute to the industry's rich heritage. Performances are delivered with a blend of earnestness and theatricality, capturing the essence of characters who are both archetypal and refreshingly unique. The lead actors navigate their roles with a finesse that suggests a deep understanding of the genre's nuances. Om Shanti Om does not merely tell a story; it invites viewers to partake in an experience that is at once familiar and novel. It challenges audiences to reflect on the cyclical nature of narratives and the enduring power of love and ambition. This cinematic endeavor provides a journey that is as thought-provoking as it is entertaining, which is not much."
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)

print("For the text:", data["text"])
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
