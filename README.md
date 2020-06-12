# Protein Folding 2D HP-Model Simulation

Protein plays a big role for every living organism, it keeps us healthy by performs it functionality such as transporting oxygen, part of imune system, muscle development, etc. These amazing functionality only works if those protein 'fold' into a stable(native)-state where it has a minimum 'free-energy' value. Predicting the minimum 'free-energy', helps bioinformatics, biotechnology in terms of developing a novel drug, vaccine, and many more. This project helps to simulate the folding process in 2D HP-model which is also well known as NP-complete problem.

## Getting Started

If you ever play around any of **OpenAI gym** environment, you should be familiar with the reset(), step(), render() function.  
Well, this project follows the **OpenAI gym** funtions behaviour too.

Before using the environment
```
git clone https://github.com/alvinwatner/protein_folding.git
```
and after done clonning, Please run the code **create_background.py** inside '**protein_folding\Code**' folder. 
It will generate an initial background '**.npy**' file for visualization purpose, otherwise it will raise an error.

![create_background](https://user-images.githubusercontent.com/58515206/84480578-f60c8200-acbe-11ea-9cc2-ad220a287f38.PNG)

### Prerequisites

* numpy               >=   1.18.2
* matplotlib          >=   3.2.1
* opencv-python       >=   4.2.0.34
* Pillow              >=   7.1.2

### Installation
Open Terminal and install the above prerequisites libraries

```
pip install numpy
pip install matplotlib
pip install opencv-python
pip install pillow
```

## How it works?

HP-model looks like a simple **board game**, since 20 different amino acid in protein are classified into 2 amino acid:
*	**‘H’ = Hydrophobic**  (Black Circle)
*	**‘P’ = Hydrophillic** (White Circle)

Given a sequence of amino **['H', 'P']**, the agent task is to place each amino in the sequence into 2D space. Note that, the next amino should be place side by side up, left, right, down from the previous amino. Repeat this process until all amino in the sequence has been placed.

**Example** :

         UP                  Left                 Right                Down

<img src="https://user-images.githubusercontent.com/58515206/84485971-eee97200-acc6-11ea-9d24-e0ed2f09b990.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84485971-eee97200-acc6-11ea-9d24-e0ed2f09b990.PNG" width="150" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84487608-32dd7680-acc9-11ea-98c1-705e57684f7b.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84488014-cadb6000-acc9-11ea-9a63-22b659dbe3cb.PNG" width="150" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84488014-cadb6000-acc9-11ea-9a63-22b659dbe3cb.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84487959-b7c89000-acc9-11ea-9a2f-259b96c88e7a.PNG" width="150" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84488248-1c83ea80-acca-11ea-937e-ad5463365637.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84488248-1c83ea80-acca-11ea-937e-ad5463365637.PNG" width="150" height="200" />

## Goals

Find the minimum total free energy given a sequence of amino acid. Free energy indicated by **H-H pairs** that is **not connected** to the protein primary structure. The value of free energy is **-1** for each pair.

**Example** :


     Free Energy = -1           Free Energy = -3             Free Energy = -9         

<img src="https://user-images.githubusercontent.com/58515206/84496947-18f76000-acd8-11ea-8aeb-9c920567a061.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84496947-18f76000-acd8-11ea-8aeb-9c920567a061.PNG" width="200" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84496527-4b548d80-acd7-11ea-8e60-62f07da85663.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84496527-4b548d80-acd7-11ea-8e60-62f07da85663.PNG" width="200" height="200" /> <img src="https://user-images.githubusercontent.com/58515206/84496565-5c9d9a00-acd7-11ea-951f-6d5e1880b76e.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84496565-5c9d9a00-acd7-11ea-951f-6d5e1880b76e.PNG" width="200" height="200" /> 

## Punishment

* Placing amino to occupied space by other amino is not allowed, it considered as **collision** and recieve a **collision punishment -2**

**Example** : 

<img src="https://user-images.githubusercontent.com/58515206/84498110-34636a80-acda-11ea-8acd-667fea5b78c4.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84498110-34636a80-acda-11ea-8acd-667fea5b78c4.PNG" width="250" height="250" />

* If amino has nowhere to go, whereas there are still other amino in the sequence, it considered as **trap condition** and receive a **trap punishment -4**

**Example** :

<img src="https://user-images.githubusercontent.com/58515206/84499172-39291e00-acdc-11ea-863f-afa4e4d5fdf0.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84499172-39291e00-acdc-11ea-863f-afa4e4d5fdf0.PNG" width="300" height="300" />

* If **collision** and **trap** occur, agent should pick another action to **move to other direction**. But there are also a conditions where the agent couldnt move to other direction since all space has occupied. If these occur, I called it as **multiple trap** then **episode terminate** (Done = True).

**Example** :

<img src="https://user-images.githubusercontent.com/58515206/84499910-bd2fd580-acdd-11ea-9f16-c78736fba429.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84499910-bd2fd580-acdd-11ea-9f16-c78736fba429.PNG" width="350" height="350" />


## Reward Function

Reward is calculated at the end of the episode, which mean its a **sparse reward RL problem**, everysteps has **0** reward except the **terminal state**

<img src="https://user-images.githubusercontent.com/58515206/84501249-64ae0780-ace0-11ea-9c48-44ba4a6b623d.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84501249-64ae0780-ace0-11ea-9c48-44ba4a6b623d.PNG" width="750" height="120" />

## Try it!
Note that, env.reset() argument is optional, if amino_input not specified then it will generate random sequences.

* env.render()            :  To Visualize Folding Process (opencv window followed by matplotlib figure)
* env.render(plot = True) :  To Show Folding Result(Matplotlib Figure) Only
```python
from simulation import environment
import numpy as np

env = environment()
current_state = env.reset(amino_input = ['P', 'P', 'P', 'H', 'H', 'P', 'P', 'H', 'H', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'P', 'P', 'H', 'H', 'P', 'P', 'P', 'P', 'H', 'H', 'P', 'P', 'H', 'P', 'P'])
done = False

while not done:
	action = np.random.randint(0, env.action_space_size)
	new_state, reward, done = env.step(action)	
	# env.render()
env.render(plot = True) #show result figure only
```
**Output**

<img src="https://user-images.githubusercontent.com/58515206/84502800-34b43380-ace3-11ea-953b-7bfe476f6c12.PNG" alt="" data-canonical-src="https://user-images.githubusercontent.com/58515206/84502800-34b43380-ace3-11ea-953b-7bfe476f6c12.PNG" width="450" height="450" />
