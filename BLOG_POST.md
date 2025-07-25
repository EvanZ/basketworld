# Can an AI Learn to Play Basketball?
### An Adventure in Reinforcement Learning and Self-Play

From the moment Deep Blue checkmated Garry Kasparov, humanity has been fascinated by the idea of artificial intelligence mastering our most complex games. We saw it again when AlphaGo made a move so alien and beautiful it was described as being "from another world." More recently, AI agents have achieved superhuman performance in intricate video games like Dota 2 and StarCraft II, not by following human instructions, but by learning on their own.

The secret to many of these breakthroughs is a powerful concept called **self-play**. You create a simulated world, build an AI agent with a blank slate, and have it play against copies of itself millions of times. Through pure trial and error, guided by a simple system of rewards and penalties, it slowly discovers the foundational principles and, eventually, advanced strategies of the game.

Inspired by this incredible history, we asked a simple question: **Could we apply the same idea to basketball?**

This is the story of BasketWorld, our project to build a custom basketball simulation and train AI agents from scratch to see if they could learn the fundamentals of the sport.

## What is Reinforcement Learning? (The 2-Minute Version)

Imagine training a puppy. You don't give it a rulebook for how to sit. Instead, when it accidentally sits, you give it a treat (a reward). If it chews on the furniture, it gets a firm "No!" (a penalty). Over time, the puppy learns which actions lead to treats and which don't.

Reinforcement Learning (RL) is the same idea for computers. Our "puppy" is a neural network called a **policy**, and its environment is a simulation.
- **Action:** The policy tries a move (like shooting a basketball).
- **Reward:** If the action helps it achieve its goal (scoring), it gets a positive reward. If it does something bad (like throwing the ball out of bounds), it gets a negative reward.
- **Learning:** The policy's goal is to maximize its total reward over time. After millions of attempts, it learns a "policy" for what actions are best in any given situation.

## Building the Arena: The BasketWorld Environment

To train an agent, we first needed a world for it to live in. We built the BasketWorld environment from the ground up using Python and the `gymnasium` library (the modern standard for RL environments). We made several key design choices:

- **A Hexagonal Court:** Instead of a simple square grid, we chose a court made of hexagons. This allows for more natural, fluid movement and six directions for passing and dribbling, which is closer to the real world than the four cardinal directions of a grid.
- **Team-Based Play:** This isn't a solo game. We have two teams—Offense and Defense—each with multiple players controlled by their own policies.
- **The Rules of the Game:** We programmed in the fundamental rules: a shot clock, a 3-point line (implicitly, through shot probabilities), turnovers for stepping out of bounds, and the core actions of moving, passing, and shooting.

![A rendered image of the BasketWorld court](placeholder_for_image.png "The BasketWorld Court")
*A visualization of our hexagonal court, with the offense in blue and the defense in red.*

## The Grand Experiment: Training with Self-Play

Our training process is a digital echo of how human athletes train: by scrimmaging. We use a technique called **alternating self-play** with a popular RL algorithm called **Proximal Policy Optimization (PPO)**, powered by the `stable-baselines3` library.

Here's how it works:
1. We initialize two AI "brains"—an Offense Policy and a Defense Policy—both as blank slates.
2. We "freeze" the Defense policy and have the Offense play against it for thousands of games. The Offense slowly learns how to score against this static opponent.
3. Then, we flip it. We freeze the newly-trained Offense policy and have the Defense play against *it* for thousands of games. The Defense learns how to stop the specific strategies the Offense has developed.
4. We repeat this cycle over and over.

The hope is that this creates an "arms race" where both sides progressively get smarter, developing more and more complex strategies to counter each other.

## The Bumps Along the Road: When AI Training Goes Wrong

The path to intelligent behavior was not a straight line. Our journey was filled with fascinating bugs and "aha!" moments that reveal the strange and counter-intuitive nature of training an AI from scratch.

#### The "Shoot Immediately" Problem
Our first successful agents learned one thing very well: how to shoot. In fact, they would shoot the ball the instant they got it, no matter where they were on the court. The episode would last exactly one second.

**The Problem:** The AI figured out that a low-probability shot was still a better bet than the risk of a turnover or the uncertainty of a pass.
**The Fix:** We had to carefully re-shape the rewards. We made long-distance shots have a near-zero chance of success and added a penalty for shooting too early in the possession. This forced the agent to learn that it needed to *work* for a better shot.

#### The Pacifist Defense
For a long time, our defensive agents wouldn't move. At all. They would stand still while the offense scored on them.

**The Problem:** The defensive policy was trying to perform illegal actions, like shooting or passing. Our training code was catching these illegal actions and overriding them to "do nothing." The agent was trying to play, but it had learned the wrong rules!
**The Fix:** We had to dive deep into our training loop and ensure that the action-masking (the system that defines what moves are legal) was being strictly enforced *before* the agent made its decision.

## Why Does This Matter? The Search for Novelty

This project is more than just a fun technical challenge. By having agents learn strategy in a simulated environment, we open the door to discovering ideas that humans might have missed. AlphaGo famously developed strategies that were new to Go players, and sports teams are increasingly using AI to analyze game data and find novel plays.

Could an AI trained in BasketWorld discover a new kind of pick-and-roll? A new defensive formation? The strategies our agents have learned so far are simple—a direct drive to the basket is their go-to move—but the foundation is there. As we continue to train and refine the environment, we are excited to see what emergent strategies might appear.

## What's Next?

Our journey with BasketWorld is just beginning. We plan to:
- Experiment with more sophisticated reward structures (e.g., rewarding "assists" or good defensive positioning).
- Scale up to a full 5-on-5 game.
- Build a web-based interactive demo where anyone can play against our trained agents.

Thanks for reading, and stay tuned for more updates from the world of AI basketball! 