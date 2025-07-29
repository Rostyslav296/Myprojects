On this Repo there are a number of Python scripts, here is a summary of them:

**Piranha.py** is a python script for launching local LLMs: deepseek r1-7b, Qwen 3, and gemini 1.5pro.
The purpose of the script is to give local LLM models agentic properties. This script is not fully functional as the models tend to hallucinate or behave in a clunky way. It is a work in progress. 
The cool thing about this script is the organiation of multiple LLM "agents" in one framework with a dedicated tools section in the script to develop. 
I ran into the issue of iterations constantly breaking the program or the boot structure, so I have clearly demarcated the boot related code, so you can edit the parts that need iteration, but leave the main boot structure in tact. 


**btcminer.py** is a python script for mining btc solo or via a pool. It is basically a lottery ticket as the complexity of btc 
is beyond profitable for mining on personal computers. 


**agent-os-v1.py** is an earlier version of Piranha.py that had a few placeholder tools along with the 3 agentic wrappers for the LLMs.
I ended up trying to keep the agents, and any python tools separate for now for easier editing, and troubleshooting. 


**Anacharssis_Z1.py** was the first agentic wrapper I made for an LLM local model that actually worked somewhat.
It can launch apps like Safari using Gui, it could make files. However it is based on Deepseek r1-7b so its still clunky
most of the tools don't work. It needs a lot of troubleshooting. To run the agent you must input the python command, the script directory
then it will list mac commands at first, just press enter and wait for the enter your query line to come up. 
unfortunately all my iterations of the Anacharssis agent don't work very smoothly and need refinement. 
There are three versions of the script, oldest: old,Anacharssis.py, then Anacharssis_Z1.py, then Anacharssis_Z2.py
Z2 is broken and not working. Functionality on the other ones is limited. 


**Compiler.py ** is an attempt to make a compiler for Mac os. I've been trying for a while now to get it to build an N64 dev kit
because I want to be able to convert .c to .z64 rom files. The build keeps failing and I've tried so many iterations,
and used Grok4, Qwen3, Kimi 2 on it to no avail so far. 


**dogecoin.py** is a dogecoin and dogecoin + litecoin python script miner. Hash rate is very slow, and its not very profitable. 
It was made just to learn more about mining. 


This repo contains a couple weeks worth of vibe coding. I'm pretty tech savvy but I don't know any programming languages. 
All the code here was vibe coded. Although I think to be a vibe coder you probably need to at least know some code. 
I will begin learning how to code to better understand these scripts I generated. I like the act of creating itself. 
There is something very satisfying about iterating a script for a long time. To see it come together at least somewht. 


