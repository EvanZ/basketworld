# Welcome to BasketWorld
## A Reinforcement Learning Environment for Simulating Basketball
---
### Motivation

As a Data Scientist over the past decade plus I have built countless machine learning models that are typically described as either supervised learning (learning from explicitly labeled examples) or unsupervised learning (learning relationships between unlabeled examples). I've built language models (albeit not "large" ones) that can be considered semi-supervised since the data provides its own labels. Recently I've become interested in reinforcement learning (RL henceforth), which can be thought of as the third major type of machine learning, and has its origins in both computational learning theory and behavioral psychology. The basic idea is that we try to learn a sequence of behaviors in order to maximize some reward given by the environment -- ok technically, we try to maximize something called "the return", which is the sum of all future rewards (often discounted in time to prioritize present over future rewards). The applications of RL are ubiquitous these days --- helping to build LLM's like ChatGPT, solving classic board games, such as Chess or Go, self-driving cars, robotics, and much more. 

So you might be asking what does any of this have to do with basketball? It's pretty simple really. I often use basketball to teach myself new algorithms or modeling techniques, because I know basketball data so well and am so motivated to do something interesting with it (thanks to my blogging and now ex-Tweeting-ex-X days). It just helps me iterate faster and have some skin in the game so to speak. So what can we do with RL and basketball? The answer is we can try to teach agents to play basketball through self-play (https://openai.com/index/competitive-self-play/). This, for example, is how DeepMind built AlphaGo and AlphaZero to become superhuman board game players. They also have used it to build soccer AI, which is super cool if you haven't seen it (https://arxiv.org/pdf/2105.12196).   

In fact my first thought along these lines was to try to build something like the soccer AI using humanoid robot physics (https://mujoco.readthedocs.io/en/stable/overview.html), but I quickly realized the complexity that would be involved and decided a more modest approach was warranted for a starting point. BasketWorld is inspired by the academic RL "playgrounds", such as "GridWorld" and "Frozen Lake" which are commonly used to test new algorithms. In these "simple" 2-D simulations, an agent is usually tasked with moving step-by-step across some, well, grid world, until it reaches an arbitrary goal, possibly collecting rewards along the way or it dies trying. Through reward "shaping" and the setting of various constraints the learned behavior of the agent can be altered and, of course, studied to the endless delight of the researcher. 

I think — I propose? I hope? — that we can do something like this for basketball, and if we learn anything from it along the way, it will be worth the effort, my effort for doing and your effort for reading about it --- but maybe you will be intrigued enough to do something along these lines too! 

### Building a World of Basket

Ok, so with my semi-short spiel out of the way, let's get to the good stuff. First, we need to think about how to build the BasketWorld. It starts with the geometry and the "court". Of course, my first thought was to use a regular square grid as in GridWorld. Squares are fine for what they are but *everyone* knows that the square walked so the hexagon could run. What is neat about the hexagon is that it is the largest regular polygon (in terms of number of sides) that can tesselate a plane completely (ie fills a plane with no gaps). (https://www.redblobgames.com/grids/hexagons/)

![hex court](../fixed_scale_basketball.png)

We are also going to start with half-court simulations, and  hence, only one basket. We can vary the number of players, the size or coarseness of the hexagons, the dimensions of the court, without much additional complexity. Next, we need to decide on the "Action Space" and the physics or rules of the simulation. First it should be said that the "atomic unit" of RL training is an "episode". For our purposes, an episode right now is defined from the start of a possession until one of these events occurs:
* a shot is attempted, regardless of whether it is missed or made (we are not modeling rebounds for now)
* a pass is stolen by a defender (I'll go into more on that in a bit)
* a bad pass is made and goes out of bounds
* the ball-handler moves out of bounds (it will become clear why that might happen)

An episode is started with the ball randomly given to one of the players on offense. Now let's define the "Action Space", which is the set of actions or "moves" each player can make during a step of the simulation. We have 14 of them to start:

* NO-OP
* MOVE_E
* MOVE_NE
* MOVE_NW
* MOVE_W
* MOVE_SW
* MOVE_S
* SHOOT
* PASS_E
* PASS_NE
* PASS_NW
* PASS_W
* PASS_SW
* PASS_SE

![Ikari Warriors](../images/ikariBox02.jpg)

I like to visualize the old Ikari Warriors arcade game which had rotary hex knobs for aiming the gun and moving around. Similarly, our players can move in any of the 6 hex directions one step or they can pass in one of said directions or shoot. They can also do nothing (NO-OP). Now there's a lot more to be said here about how we model the physics of these actions. For example, what do we do if two players try to move into the same hex cell? Well, as god (king? Prime Minister?) of BasketWorld, I decided to disallow such collisions. This makes the sim much easier to reason about and for example makes implementing defense simpler since "defending" a position means simply occupying it. That seems reasonable (for now, at least). 

The passing "physics" is quite simple. Once the direction is chosen we just look for the first player occupying a hex cell in that direction. If it is a teammate the ball is caught (of course, we can easily add some chance that it is turned over and ends the episode). If the first player in the "line of sight" (LOS) is a defender, then the pass is "stolen" and the episode ends. If no player is in the LOS the ball "goes out of bounds" and the episode ends. 

If the chosen action is "Shoot", the player makes a shot attempt with some fairly simplistic linear probability distribution depending on distance from the basket. I'm not even going to bother giving a formula here because it will probably change countless times before I even publish this and certainly when you are reading it. These kinds of choices essentially get to the heart of the whole BW enterprise. You can imagine down the road even building in player-specific probability distributions. There's so much that can be done...I'm sure you can see it too.

Ok, now with our world essentially built, how do our basketball agents learn how to play? Notice --- and this is really important to internalize --- while we have specified the types of actions players can make, nowhere are we going to explicitly tell them what to do...the idea of RL and in this case, specifically, the idea of self-play RL, is that the agents will learn how to play by repeated episodes playing against each other. So let's talk about how that works.

### Training and Rewards

Before getting into the details of how BasketWorld agents are trained, let's step back and give an overview of how RL works in general. Without getting too technical (again see the Sutton and Barto book for the nitty gritty details and theory), the main idea is that given some reward system, we "roll out" episodes where an agent initially performs completely random actions sampled from the action space. For example, in Grid World the actions are simply moves in each of 4 directions. Most of the time the agent won't receive an award but sometimes they will receive a positive reward and in that case the various RL algorithms (there are many) try to assign or attribute credit to the sequence of actions that lead to receiving the reward. In the simplest RL scenarios like Grid World, you can literally tabulate using a lookup table how often a certain action resulted in a positive (or negative) reward down the line. 

Now of course, just sampling actions at random forever would be wildly inefficient and not ever lead to learning a regular behavior. (Not even Lex Luthor could employ enough typing monkeys to write Hamlet.) No, the point of all this is to create a "policy" which our agent can use to select the best action in order to maximize its reward from the environment. So as we run more and more episodes we start to feedback reward attribution into our policy leading to improvements until we reach some optimal policy. With this policy we can then...have self-driving cars or computer programs that can beat anyone at Chess. 

Or even basketball. So let's get back to BasketWorld and how we use this specifically. In BasketWorld we have 2 agents: Offense and Defense. Each team has multiple players but those players are not really independent agents, they are carrying out actions according to either an offensive or defensive policy, which is what we try to learn through self-play. We alternate training offense and defense. Each "team" starts with a random policy. We start by training the offense and improving that policy. So let's say we perform 20K episodes and the offense has gotten a little bit better. We then "freeze" that offensive policy, and train the defense for 20K episodes. Then we freeze the defensive policy, and train the offense against it. We keep alternating in this way as many times as we need to. It's very much like the saying "iron sharpens iron". Both sides are getting better by learning how to gain rewards against the other side.

Ah, and I've buried the lede. I haven't told you what the rewards in this game actually are. Well in some sense I've saved the best and most interesting aspect of the whole endeavor for last because it warrants a fairly lengthy discussion. Of course the obvious reward in basketball is a win or a loss, but we are trying to model a single possession (sans rebounding), which makes the obvious reward a made basket. So when we are training offense let's say the reward for a made basket is +1 (obviously down the line we can model a 3pt line) and anything else that ends the episode simply results in no reward. On defense we do the mirror opposite. When the defense gives up a basket the reward is -1. When the episode ends any other way there is no reward. With these rewards and whatever actions and "physics" and rules we have specified, the two sides will play each other trying to maximize rewards on offense and minimize opponent reward on defense, exactly like a real game. 

> This is where an RL researcher invariably mentions the cautionary warning about some future "tidy bot" that given instructions to keep the office clean, decides humans are the cause of uncleanliness and proceeds to gun down everyone in the building ED-209 style.  

So quite literally the agents during training attempt to maximize reward mercilessly. Whatever strategy leads to optimal reward will be employed even if it is not what you had in mind. In my first attempts to train BasketWorld I noticed that the average episode length always converged to 1 step. What was happening? Was there a bug in my code? Nope, the agent was simply learning that the best strategy was to immediately shoot the ball from anywhere on the court rather than pass or do anything else. Most likely because extending an episode would just give a higher probability of a turnover due to some random pass action. With this in mind I tried my first "reward shaping" experiment just to convince myself that there wasn't a bug and this whole BasketWorld idea wasn't completely pointless. What I wanted was some way to extend an episode beyond 1 step (and keep in mind we have a shot clock where each "second" represents a step). To do this I instituted a linearly decaying negative reward which was greater at the beginning of the shot clock than the positive reward for making a basket. I should note that this turns our initial "sparse" reward (only given on made baskets) into a "dense" reward. Well the effect of this new negative reward was indeed that it extended episodes by several seconds for 1 to roughly 6 or 7 steps. It turned out that the strategy learned by the agent under this new policy is always to drive immediately to the basket and then shoot when it gets to the "rim". A few illustrative episodes are shown below (the ball handler is the blue circle with the orange ring around it).

![driving gif 1](../made_shot_viz.gif)

![driving gif 2](../test_reward.gif)

Our agent makes a basket nearly 80% of the time, which is not surprising given our shot probability distribution looks like this:

```
    def _calculate_shot_probability(self, shooter_id: int, distance: int) -> float:
        """Calculate probability of successful shot based on distance."""
        if distance <= 1:
            return 0.9  # Dunk/Layup
        elif distance <= 3:
            return 0.5  # Close shot
        elif distance <= 5:
            return 0.2  # Mid-range
        else:
            return 0.05 # Long-range heave
```

The optimal policy being learned makes perfect sense given our rewards and the constraints/physics of the environment. The offense doesn't always win though. Here is an episode where a defender (#3 red) puts the ball handler (#1 blue) in prison (yes you have to use your imagination here):

![great defense](../mlruns/101107429080230513/ab0f402bc060442fb669c60f696af773/artifacts/gifs/missed_shot_viz.gif)

So even in these very early days I'm encouraged that we are seeing something resembling game logic. There's so much more to do and learn here. Going forward here are examples of topics we can explore in BasketWorld:

* What is the simplest reward system and game physics (if any) that leads to policies which resemble real NBA actions, such as pick and roll or backdoor cuts? 
* What happens when we institute a 3pt line?
* If we vary the skill levels (shooting, turnover likelihood, etc) of individual players can we learn policies that take advantage of it? 
* Instead of using grid-based discrete movement physics, can we model continuous player trajectories and velocities?
* How do we model fouls?

These are just a few areas off the top of my head. I'd love to hear your suggestions in the comments for future BasketWorld experiments! 

### Technical Details

For those interested in the technical details, BasketWorld is built on the Gymnasium standard, with agents trained using the PPO algorithm from the Stable-Baselines3 (https://stable-baselines3.readthedocs.io/en/master/#) library and experiment tracking managed by MLflow. The project is open source on github under a MIT license. If you like the project please support it by giving it a star. 

https://github.com/EvanZ/basketworld