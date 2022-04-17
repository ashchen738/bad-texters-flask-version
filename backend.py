# install nltk
# install spacy
# overall tasks to accomplish
# 1. parse the json to split into the two individuals
# 2. sentence analysis
# 3. create a few metrics of success (maybe start with how good of a texter they are)
# things to analyze
# 1. sentiment analysis:
#       - passive aggressiveness
#       - anger?
#       - no emotion
# - emojis
# 2. response time - done
# 3. most commonly used words (done) - could also use for like how apathetic people are? - idt it would b helpful
# 4. sent + received ratio - done
# 5. activity (how active they are - start, most recent, streak, av freq)
# 6. how often do they initiate conversations - nope
# 7. how often they leave ppl on read - done

import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from heapq import heappush, heapify, heappop
stopwords = []
with open("stop_words_english.txt") as f:
    for line in f:
        line = line.strip()
        stopwords.append(line.lower())


def read_file(filename):
    with open(filename) as f:
        jsonfile = json.load(f)
        json_content = jsonfile["messages"]
        li = []
        json_participants = jsonfile["participants"]
        recip = json_participants[0]["name"]
        send = json_participants[1]["name"]
        #recip, send = "", ""
        for i in json_content:
            content = ""
            if "content" in i:
                content = i["content"]
            reaction = False
            if 'Reacted' in content and 'to your message' in content or "reactions" in i:
                reaction = True
            li.append((i["sender_name"], i["timestamp_ms"], content, reaction))
    return li[::-1], recip, send


def reaction_time(li):
    curr_recipient = li[0][0]
    reaction_times = {}
    for i in range(1, len(li)):
        if li[i][0] != curr_recipient and i != 0:
            response_time = li[i][1] - li[i-1][1]
            curr_recipient = li[i][0]
            if curr_recipient not in reaction_times:
                reaction_times[curr_recipient] = [response_time, 1]
            else:
                reaction_times[curr_recipient][0] += response_time
                reaction_times[curr_recipient][1] += 1
    for i in reaction_times:
        reaction_times[i] = reaction_times[i][0]/reaction_times[i][1]
    return reaction_times


def most_used(li, recipient, sender):
    curr_recipient = li[0][0]
    curr_recipient = sender
    li_participant1 = {}
    li_participant2 = {}
    for i in li:
        if i[-1] == False:
            words = i[2].split(" ")
            for w in words:
                if i[0] == curr_recipient:
                    if w in li_participant1:
                        li_participant1[w] -= 1
                    else:
                        li_participant1[w] = -1
                elif w in li_participant2:
                    li_participant2[w] -= 1
                else:
                    li_participant2[w] = -1
    li1 = [(value_, i) for i, value_ in li_participant1.items()
           if i.lower() not in stopwords and i != ""]

    li2 = [(value, i) for i, value in li_participant2.items()
           if i.lower() not in stopwords and i != ""]

    heapify(li1)
    heapify(li2)
    # print(li1)
    top_30_1 = []
    top_30_2 = []
    for i in range(30):
        top_30_1.append(heappop(li1))
        top_30_2.append(heappop(li2))
    return top_30_1, top_30_2

#[(Name, time, reaction)]
# jsoncont = read_file("test/inbox/meghanagopannagari_r2imp6gtla/message_1.json")
# print(reaction_time(jsoncont))

# maybe do a thing like take the first response time(maybe like 5 seconds)
# and add +=1/4 of that time (if they continue to stay in that interval) they're still part of the convo


# returns how many times they left someone on read
def leaving_on_read(li, recipient, sender):
    time = 1000*60*60*3  # three hour for now
    # if the same person texts twice in a row with more than 3 hours in between they got left on read :((
    # another one is like long reaction time but that's already factored I think?
    curr_recipient = li[0][0]
    reaction_times = {}
    for i in range(len(li)):
        if li[i][0] == curr_recipient and i != 0:
            response_time = li[i][1] - li[i-1][1]
            if response_time > time:
                if curr_recipient in reaction_times:
                    reaction_times[curr_recipient] += 1
                else:
                    reaction_times[curr_recipient] = 1
                # they had to double text --> got left on read :((
        else:
            curr_recipient = li[i][0]
    a = reaction_times[curr_recipient]
    other = [i for i in [sender, recipient] if i != curr_recipient][0]
    reaction_times[curr_recipient] = reaction_times[other]
    reaction_times[other] = a
    return reaction_times


def isolate_block(li, name):  # provide name (sender or recipient)
    curr_recipient = li[0][0]
    curr_pos = 0
    curr_block = ""
    blocks1 = []
    for i in range(len(li)):
        if li[i][0] != curr_recipient and i != 0 and li[i][1] - li[curr_pos][1] >= 10800000:
            curr_pos = 0
            if curr_recipient == name and curr_block != "" and "to your message" not in li[i][2]:
                blocks1.append(curr_block)
            curr_block = ""
            curr_recipient = li[i][0]
        else:
            curr_block += f'{li[i][2]}, '

    return blocks1


def sentiment(li):  # list of blocks by the sender
    nltk.download('vader_lexicon')
    model = SentimentIntensityAnalyzer()
    li_sentiments = [model.polarity_scores(i)["compound"] for i in li]
    return sum(li_sentiments)/len(li_sentiments)


# how many times do they initiate convo --> so a while loop from the start where its like 3 hours apart?
# in a conversation, how man
# do this later:
"""
def conversation_initiation(li):
    time = 1000*60*60*3
"""


def reaction_counter(li, name):  # number of reactions
    num_react = 0
    for i in li:
        content = i[2]
        if name == i[0] and i[-1] == True:
            num_react += 1
    return num_react


def count_sent(li, recipient, sender):
    count = 0
    for i in li:
        name = i[0]
        if(name == sender):
            count += 1
    return count


def count_received(li, recipient, sender):
    count = 0
    for i in li:
        name = i[0]
        if(name == recipient):
            count += 1
    return count


def calc_diff(li, recipient, sender):
    return count_sent(li, recipient, sender) - count_received(li, recipient, sender)


def avg_length(li, name):
    count = 0
    total_len = 0
    for i in li:
        if i[0] == name:
            current_len = len(i[2])
            count += 1
            total_len += current_len
    return total_len / count


def compile_all(li, recipient, sender):
    reactt_1, reactt_2 = reaction_time(
        li)[sender], reaction_time(li)[recipient]
    # print(reactt_1)
    words_1, words_2 = most_used(li, recipient, sender)[
        0], most_used(li, recipient, sender)[1]
    # print(words_1[0])
    read_1, read_2 = leaving_on_read(
        li, recipient, sender)[sender], leaving_on_read(li, recipient, sender)[recipient]
    # print(read_1)
    isolate1, isolate2 = isolate_block(
        li, sender), isolate_block(li, recipient)
    # print(isolate1)
    senti1, senti2 = sentiment(isolate1), sentiment(isolate2)
    # print(senti1)
    reactions1, reactions2 = reaction_counter(
        li, sender), reaction_counter(li, recipient)
    # print(reactions1)
    texting_dif1, texting_dif2 = calc_diff(
        li, recipient, sender), calc_diff(li, recipient, sender)*-1
    # print(texting_dif1)
    avlen_1, avlen_2 = avg_length(
        li, sender), avg_length(li, recipient)
    # print(avlen_1)
    dict1 = {"react": reactt_1, "common_words": words_1, "left_on_read": read_1, "sentiment": senti1,
             "num_reactions": reactions1, "texting_dif": texting_dif1, "avg_length": avlen_1}
    dict2 = {"react": reactt_2, "common_words": words_2, "left_on_read": read_2, "sentiment": senti2,
             "num_reactions": reactions2, "texting_dif": texting_dif2, "avg_length": avlen_2}
    return dict1, dict2


def calc_score(li, recipient, sender):
    dict1, dict2 = compile_all(li, recipient, sender)
    sen1_score = dict1["sentiment"] * \
        200 if(dict1["sentiment"] >= 0.5) else (1 - dict1["sentiment"]) * 200
    sen2_score = dict2["sentiment"] * \
        200 if(dict2["sentiment"] >= 0.5) else (1 - dict2["sentiment"]) * 200
    text1_score = (1 - (abs(dict1["texting_dif"]) /
                   (count_received(li, recipient, sender)))) * 100
    text2_score = (
        1 - (abs(dict2["texting_dif"]) / (count_sent(li, recipient, sender)))) * 100
    read1_score = (abs(len(isolate_block(li, sender)) -
                   dict1["left_on_read"])) / len(isolate_block(li, sender)) * 100
    read2_score = (abs(len(isolate_block(li, recipient)) -
                   dict1["left_on_read"])) / len(isolate_block(li, recipient)) * 100
    time1_score = (86400000 - dict1["react"]) / 86400000 * \
        100 if(dict1["react"] < 86400000) else 0
    time2_score = (86400000 - dict2["react"]) / 86400000 * \
        100 if(dict2["react"] < 86400000) else 0
    # / (count_received(isolate_block(li, sender))) * 100
    react1_score = dict1["num_reactions"]
    # / (count_sent(isolate_block(li, sender))) * 100
    react2_score = dict2["num_reactions"]
    sender_score = (0.3 * sen1_score) + (0.2 * text1_score) + \
        (0.1 * read1_score) + (0.1 * time1_score) + (0.3 * react1_score)
    recipient_score = (0.3 * sen2_score) + (0.2 * text2_score) + \
        (0.1 * read2_score) + (0.1 * time2_score) + (0.3 * react2_score)
    dict1_scores = {"sen1_score": sen1_score, "text1_score": text1_score,
                    "read1_score": read1_score, "time1_score": time1_score, "react1_score": react1_score}
    dict2_scores = {"sen2_score": sen2_score, "text2_score": text2_score,
                    "read2_score": read2_score, "time2_score": time2_score, "react2_score": react2_score}
    return sender_score, recipient_score, dict1_scores, dict2_scores


# jsonc, recep, send = read_file("message_1val.json")
# print(calc_score(jsonc, recep, send))
# print("\u00e2\u009d\u00a4")
# print(reaction_time(jsoncont))
# print(most_used(jsoncont)[1])
# print(leaving_on_read(jsoncont))
# print(stopwords)
#print(isolate_block(jsoncont, sender)[0], isolate_block(jsoncont, recipient)[0])
#print(reaction_counter(jsoncont, sender))
