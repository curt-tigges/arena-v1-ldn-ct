import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
from datetime import datetime
import re
import pandas as pd
import platform
is_local = (platform.processor() != "")
rootdir = "" if is_local else "streamlit/ch1/"

st.set_page_config(
    # initial_sidebar_state="expanded",
    layout="centered"
)

def generate_fig():
    f = rootdir + "images/headers/"
    f_table = rootdir + "images/table.csv"
    datetime_index = pd.date_range(start="2022-10-31", periods=7*10)
    datetime_index = datetime_index[datetime_index.dayofweek <= 4]
    df = pd.read_csv(f_table, keep_default_na=False)
    df = df[[i != "" for i in df["Chapter"]]]
    days_per_section = []
    current_chapter = ""
    for chapter in df["Chapter"].values:
        if chapter != current_chapter:
            days_per_section.append(0)
            current_chapter = chapter
        days_per_section[-1] += 1
    weekday_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    arr_list = []
    today = datetime.today().date()
    # today = datetime.strptime("28/10/2022", "%d/%m/%Y").date()
    counter = -1
    path_list = ["pre", "trans", "scale", "opti", "mod", "rl", "laws", "int", "adv", "cap"]
    for chapter_no, days in enumerate(days_per_section):
        path = f + f"{path_list[chapter_no]}.png"
        img_true = np.asarray(Image.open(path).convert('RGB').resize((164, 164)))
        img = 255 * np.ones((180, 180, 3))
        img[8:-8, 8:-8, :] = img_true
        for d in range(days):
            counter += 1
            if (
                (datetime_index[counter].date() <= today #type: ignore
                and ((counter == len(datetime_index)) or (datetime_index[counter+1].date() > today))) #type: ignore
                or (today <= datetime_index[0].date() and counter == 0) #type: ignore
            ):
                img2 = img.copy()
                img2[:, :, 0] = img2.mean(axis=-1)
                img2[:, :, 1:] = 0
                arr_list.append(img2)
            else:
                arr_list.append(img)
    for i in range(50 - len(arr_list)):
        arr_list.append(255 * np.ones((180, 180, 3)))
    arr = np.stack(arr_list).astype(int)
    b1b2, h_, w_, c_ = arr.shape
    b2 = 5
    b1 = b1b2 // b2
    arr = arr.reshape((b1, b2, h_, w_, c_))
    arr = np.moveaxis(arr, [0, 1, 2, 3, 4], [2, 0, 1, 3, 4])
    arr = arr.reshape((b2*h_, b1*w_, c_))
    fig = px.imshow(arr, zmin=0, zmax=255)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [90 + 180*i for i in range(10)],
            ticktext = [f"W{i}<br>{str(datetime_index[5*i].date())}" for i in range(10)]), #type: ignore
        yaxis = dict(
            tickmode = 'array',
            tickvals = [90 + 180*i for i in range(5)],
            ticktext = [day[:3] + " " for day in weekday_list]),
        margin=dict(t=0, b=20, r=0, l=0)
    )
    fig.update_traces(hovertemplate=None, hoverinfo="skip")

    df = pd.read_csv(f_table, keep_default_na=False)
    color_list = px.colors.qualitative.Pastel1[:-1] + px.colors.qualitative.Pastel1[:2]
    def get_color(i):
        srch = re.search(r"\d", i)
        return (int(srch[0]), color_list[int(srch[0])]) if srch else (10, "rgba(180, 180, 180, 0.25)")
    def style_func(s, column):
        return [f'background-color: {get_color(s.loc[column][0])[1]}' for _ in range(3)]
    df["Day"] = [f"<a href='{url}'>{day}</a>" for url, day in zip(df["Exercises link"], df["Day"])]
    df = df[["Date", "Day", "Chapter"]]
    table = (df.style
        .apply(style_func, column=["Chapter"], axis=1)
        .set_table_styles([
            {"selector": "td", "props": "font-weight: bold"},
            {"selector": "tr", "props": "line-height: 0.9em;"},
            {"selector": "td,th", "props": "padding: 8px;"}
        ])).to_html(escape=False)
    return fig, table

if "fig_table" not in st.session_state:
    fig, table = generate_fig()
    st.session_state["fig_table"] = (fig, table)
else:
    fig, table = st.session_state["fig_table"]

st.markdown("""
<style>
.row_heading.level0 {display:none}
.blank {display:none}
td {
    color: black !important;
}
table {
    width: calc(100% - 30px);
    margin: 15px
}
[data-testid="stDecoration"] {
    background-image: none;
}
div.css-fg4pbf [data-testid="column"] {
    box-shadow: 4px 4px 10px #ccc;
    padding: 15px;
}
div.css-ffhzg2 [data-testid="column"] {
    background: #333;
    padding: 15px;
}
[data-testid="column"] a {
    text-decoration: none;
}
code {
    color: red;
    font-size: 0.9em;
}
.highlight-reading {
    color: white;
    background-color: blue;
    border-radius: 15px;
    padding: 5px;
}
a:hover {
    text-decoration: underline;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
.css-ffhzg2 code:not(pre code) {
    color: orange;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
pre code {
    font-size:13px !important;
    line-height: 13px !important;
}
.katex {
    font-size:17px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -10px;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
</style>""", unsafe_allow_html=True)


st.markdown("""
# Curriculum

The symbol at the end of each line links to the material for that day, but it also indicates what type of day it will be: 📝 for **exercises**, 📚 for **reading**, and 🔬 for **open-ended projects/investigations**.

You can click on the tab headers below to navigate through the chapters, and click on the title of each day to be redirected to that day's material.

You can also see a <u>calendar view</u> and a <u>daily view</u> in the dropdowns immediately below. The calendar view provides a nice visualisation of the whole programme, and the daily view should help you quickly find the exercises for a certain day.
""", unsafe_allow_html=True)

with st.expander("Calendar view"):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("Daily view"):
    st.write(table, unsafe_allow_html=True)
    st.markdown("Note that this plan has some flexibility built-in. We indend to wrap up at the end of the final week before Christmas, and we might still redistribute some material between weeks depending on how the curriculum goes.")

tabs = st.tabs([f"CH {i}" for i in range(10)])

with tabs[0]:

    st.image(rootdir + "images/headers/pre.png", width=250)
    st.subheader("Chapter 0 - Prerequisites")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 5 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 Before embarking on this curriculum, it is necessary to understand the basics of deep learning, including basic machine learning terminology, what neural networks are, and how to train them.

This week concludes with you building and finetuning your own Residual Neural Network based on the **ResNet34** architecture, to classify images from ImageNet.""")

    st.info("""
📜 This chapter's material is primarily based on the first week of MLAB2. It also draws on several PyTorch tutorials.
""")


    ch1_columns = st.columns(1)
    with ch1_columns[0]:
        st.markdown("""<h5><code>W0D1</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;<a href="https://arena-ldn-ch0.streamlit.app/W0D1_-_Fourier_Transforms">Fourier Transforms 📝</a></h5>""", unsafe_allow_html=True)
        st.markdown("""
    Get comfortable with the basics of how exercises work, via an implementation of Fourier transforms. Then build a very basic neural network from the ground up, just to get an idea of what role all the different PyTorch components play.

    ---
    """)
        st.markdown("""<h5><code>W0D2</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;<a href="https://arena-ldn-ch0.streamlit.app/W0D2_-_as_strided,_convolutions_and_CNNs">as_strided, convolutions and CNNs 📝</a></h5> """, unsafe_allow_html=True)
        st.markdown("""
    Learn about `as_strided`, as well as `einops` and `einsum` - important libraries for expressing more complicated linear operations within neural networks. Then apply this knowledge to build your own Linear and Convolutional layers, which inherit from `nn.Module`. 

    ---
    """)
        st.markdown("""<h5><code>W0D3</code> & <code>W0D4</code>&emsp;|&emsp;<a href="https://arena-ldn-ch0.streamlit.app/W0D3_-_ResNets_and_fine-tuning">ResNets and fine-tuning 📝</a></h5> """, unsafe_allow_html=True)
        st.markdown("""
    Apply the lessons from the previous day, to assemble and train a CNN out of layers that you built yourself. Use it to classify MNIST data. Then, build a more complicated architecture (ResNet34) and fine-tune it on ImageNet data.

    Today's exercises are expected to run over into part of tomorrow.

    ---
    """)
        st.markdown("""<h5><code>W0D4</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;<a href="https://arena-ldn-ch0.streamlit.app/W0D4_-_Weights_and_Biases">Weights and Biases 📝</a></h5> """, unsafe_allow_html=True)
        st.markdown("""
Today, you'll be introduced to **Weights and Biases**, a tool for logging and efficient hyperparameter search. You should spend the morning on W0D3, since the associated exercises here should only take an afternoon.

---
    """)
        st.markdown("""<h5><code>W0D5</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;<a href="https://arena-ldn-ch0.streamlit.app/W0D5_-_Build_Your_Own_Backprop_Framework">Build Your Own Backpropagation Framework 📝</a></h5>""", unsafe_allow_html=True)
        st.markdown("""
Today, you'll learn about the nuts and bolts of implementing backpropagation: how gradients are stored, and how they're propagated backwards through a computational graph.

This is bonus content, and won't be essential for any other parts of the course. You may wish to return here after the course has finished.
""")

with tabs[1]:

    st.image(rootdir + "images/headers/trans.png", width=250)
    st.subheader("Chapter 1 - Transformers")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 7 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 The **transformer** is an important neural network architecture used for language modelling.

In this week, you will learn all about transformers - how the **self-attention mechanism** works, how transformers are trained, and how they've managed to be the driving force behind language model progress of the last few years.""")

    st.info("""
📜 This chapter's material is primarily based on week 1 of Jacob Hilton's curriculum. It also draws on elements from MLAB2 W2, and Marius Hobbhahn's [Building a transformer from scratch](https://www.lesswrong.com/posts/98jCNefEaBBb7jwu6/building-a-transformer-from-scratch-ai-safety-up-skilling) challenge.
""")

    ch1_columns = st.columns(1)
    with ch1_columns[0]:
        st.markdown("""<h5><code>W1D1</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;<a href="https://arena-ldn-ch1.streamlit.app/">Transformer reading & exercises 📚</a></h5>""", unsafe_allow_html=True)
        st.markdown("""
Read about transformers: the basics of their architecture, what self-attention is, how tokenisation works, etc. There are also some questions to work through, to check how well you've understood the concepts.

You'll also be going through some atomic transformer exercises: building an attention function, and a positional encoding function.

---
    """)
        st.markdown("""<h5><code>W1D2</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;<a href="https://arena-ldn-ch1.streamlit.app/">Build your own transformer (1/2) 📝</a></h5> """, unsafe_allow_html=True)
        st.markdown("""
Build your own transformer! This will be the most challenging and open-ended task you've done so far in this programme. You will also test your transformer by making it learn a simple task: reversing the order of a sequence of digits.

---
    """)
        st.markdown("""<h5><code>W1D3</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp;<a href="https://arena-ldn-ch1.streamlit.app/">Build your own transformer (2/2) 📝</a></h5> """, unsafe_allow_html=True)
        st.markdown("""

You'll now train your transformer to do a much harder task: autoregressive text generation, from training on the entire [Shakespeare text corpus](https://www.gutenberg.org/files/100/100-0.txt). To do this well, you'll also need to learn about different sampling techniques.

---
    """)
        st.markdown("""<h5><code>W1D4</code> - <code>W2D5</code>&emsp;|&emsp;<a href="https://arena-ldn-ch1.streamlit.app/">Build your own transformer (2/2) 📝</a></h5> """, unsafe_allow_html=True)
        st.markdown("""
The rest of this chapter will be spent on additional transformer exercises, including building and using GPT-2 and BERT, and trying to build a classifier using only modules you've created yourself (in a throwback to our work from week 0 assembling ResNet34). 

---
    """)

with tabs[2]:

    st.image(rootdir + "images/headers/scale.png", width=250)
    st.subheader("Chapter 2 - Training at Scale")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 5 days
    </div>
    """, unsafe_allow_html=True)

    st.markdown("This chapter is part of an optional track: you can choose to do it, or chapter 4A (modelling objectives).")

    st.success("""
💡 There are a number of techniques that are helpful for training large-scale models efficiently. Here, we will learn more about these techniques and how to use them.""")

    st.info("""
📜 It's currently unclear exactly what this week will look like, although it will probably draw at least in part from [week 3 of Jacob Hilton's curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/3-Training-at-Scale.md).
""")

with tabs[3]:

    st.image(rootdir + "images/headers/opti.png", width=250)
    st.subheader("Chapter 3 - Optimisation")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 2 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 It's helpful to have an intuition for how SGD and its variants optimize models, and a number of theoretical pictures are informative here.

We will read some papers discussing some of the mathematical justifications behind different optimisation algorithms and learning rate schedules, and conclude by running our own set of experiments.""")

    st.info("""
📜 This chapter will be designed by the ARENA team, and will also draw heavily on [week 4 of Jacob Hilton's curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/4-Optimization.md).
""")
    ch1_columns = st.columns(1)
    with ch1_columns[0]:
        st.markdown("""<h5><code>W3D3</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp; Optimisers: exercises 📝</h5>""", unsafe_allow_html=True)
        st.markdown("""Learn about different optimisation algorithms (e.g. **RMSProp** and **Adam**), and implement them from scratch. Understand important concepts like momentum, and how they affect the performance of optimisers.

---
    """)
        st.markdown("""<h5><code>W3D3</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp; Optimisers: investigations 🔬</h5> """, unsafe_allow_html=True)
        st.markdown("""Run your own experiments on optimisation algorithms. There are several different experiments you can choose to run, based on the material provided in Jacob Hilton's curriculum.""")

with tabs[4]:

    st.image(rootdir + "images/headers/mod.png", width=250)
    st.subheader("Chapter 4 - Modelling Objectives")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 4 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
Here, we take a tour through various generative models. This is the name for a broad class of models which can generate new data instances from a particular distribution. Examples are diffusion models like DALL-E 2, which we used to generate the images you're seeing on these pages!""")

    st.info("""
📜 This chapter is primarily based on [week 5 of Jacob Hilton's curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/5-Modelling-Objectives.md). It also draws on some material from week 3 of MLAB2.
""")

    ch1_columns = st.columns(1)
    with ch1_columns[0]:
        st.markdown("""<h5><code>W3D5</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp; GANs and VAEs 📝</h5>""", unsafe_allow_html=True)
        st.markdown("""
Learn how **GANs** (Generative Adversarial Models) and **VAEs** (Variational Autoencoders) work, and build & train some of your own.

---
    """)
        st.markdown("""<h5><code>W4D1</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp; Diffusion models 📚</h5> """, unsafe_allow_html=True)
        st.markdown("""
Read up on the maths behind diffusion models, and why they work so well for image generation.

Note - this day will be fairly math-heavy, so you might want to jump straight to the material following this, which will have you build your own diffusion models.

---
""")
        st.markdown("""<h5><code>W4D2 - W4D3</code>&emsp;&nbsp;|&emsp; Implementing diffusion models 📚</h5> """, unsafe_allow_html=True)
        st.markdown("""
Over these two days, you'll be implementing your own diffusion models. Like during the transformer exercises, you can choose from several different degrees of difficulty.
""")

with tabs[5]:

    st.image(rootdir + "images/headers/rl.png", width=250)
    st.subheader("Chapter 5 - RL")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 7 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 Reinforcement learning is an important field of machine learning. It works by teaching agents to take actions in an environment to maximise their accumulated reward.

In this chapter, you will be learning about some of the fundamentals of RL, and working with OpenAI's Gym environment to run your own experiments.""")

    st.info("""
📜 This chapter is primarily based on pre-existing RL tutorials, such as OpenAI's spinning up course.
""")

with tabs[6]:

    st.image(rootdir + "images/headers/laws.png", width=250)
    st.subheader("Chapter 6 - Scaling Laws")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 3 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 Studying how properties of networks **vary with scale** is important for drawing generalizable conclusions about them.

In this week, we will read foundational papers on scaling laws, and perform our own study of scaling laws for the MNIST classifiers we wrote in week 0.""")

    st.info("""
📜 This chapter's material is primarily based on [week 2 of Jacob Hilton's curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/2-Scaling-Laws.md).
""")

    ch1_columns = st.columns(1)
    with ch1_columns[0]:
        st.markdown("""<h5><code>W2D3</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp; Scaling Laws: reading 📚</h5>""", unsafe_allow_html=True)
        st.markdown("""
---
    """)
        st.markdown("""<h5><code>W2D4</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp; Scaling Laws: exercises 📝</h5> """, unsafe_allow_html=True)
        st.markdown("""
---
    """)
        st.markdown("""<h5><code>W2D5</code>&emsp;&emsp;&emsp;&emsp;&emsp;|&emsp; Scaling Laws: investigations 🔬</h5> """, unsafe_allow_html=True)
        st.markdown("""

    """)

with tabs[7]:

    st.image(rootdir + "images/headers/int.png", width=250)
    st.subheader("Chapter 7 - Interpretability")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 7 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 Mechanistic interpretability aims to reverse-engineer the weights of neural networks into human-understandable programs. It's one of the most exciting and fastest-growing fields in AI safety today.

In this chapter, you will be performing your own interpretability investigations, including feature visualisation and attribution for CNNs, and transformer interpretability exercises.""")

    st.info("""
📜 This chapter is primarily based on Anthropic's [Transformer Circuits](https://transformer-circuits.pub/) work, and material from week 2 of MLAB2. It also draws from [week 8 of Jacob Hilton's curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/8-Interpretability.md).
""")

with tabs[8]:

    st.image(rootdir + "images/headers/adv.png", width=250)
    st.subheader("Chapter 8 - Adversarial Training")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 3 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 Adversarial training is designed to make models robust to adversarially-selected inputs.

In this chapter, we will be working with the language models we've studied in previous weeks, and trying to red-team them by producing examples of offsenive language or other failures.""")

    st.info("""
📜 This chapter is primarily based on [week 9 of Jacob Hilton's curriculum](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/9-Adversarial-Training.md).
""")

with tabs[9]:

    st.image(rootdir + "images/headers/cap.png", width=250)
    st.subheader("Chapter 9 - Capstone Projects")

    st.markdown("""
    <div style="color:gray; margin-top:-30px">
    Duration: 7 days
    </div>
    """, unsafe_allow_html=True)

    st.success("""
💡 We will conclude this program with capstone projects, where you get to dig into something related to the course. This should draw on much of the skills and knowledge you will have accumulated over the last 9 weeks, and serves as great way to round off the program!""")



