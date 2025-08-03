# Creating interactive tools for understanding policy learning and game play in BasketWorld

## Using a game-within-the-game to learn the game
---
### Challenges of Reinforcement Learning and Self-Play

In general it is fairly straightforward to analyze the performance of "traditional" machine learning models, such as classification or regression models, which fall under the broader label of supervised learning. With these models there is a specific target and you know the model you have built is good if it can predict that target at some expected rate. While performance is obviously relative to the task context --- for example, you might have very different expectations of performance for an NFL wins prediction model compared to a model that predicts whether a picture contains a cat or a dog --- the actual analytics used and understanding of performance in each case is fairly straightforward. If you are building a classificaition model, you probably have a handful of "go to" metrics to evaluate performance, such as AUC, accuracy, TPR, FPR, F1, etc. And for a regression model --- depending on the specific loss function --- you probably look at RMSE, log loss, MAE, R2, etc. These metrics become very useful when you're trying to optimize performance or build new models in the same domain, because these metrics directly correspond with whatever the "real world" application is. 

While in theory RL models intrinsically have some "reward" that can be monitored and optimized for, it isn't at all as straightforward as the case for supervised learning especially when we start talking about self-play applications. The Deep RL models that were built to beat Atari games in many cases can be evaluated in a direct way by simply looking at the points the agent gained during game play. An agent playing Space Invaders or whatever gains rewards (ie points) by shooting things. An LLM such as ChatGPT is trickier to evaluate, though, right? This is why AI researchers have built tons of "real world" benchmarks to test their models. And as someone who uses these models every day, I can't even tell you what the latest metrics are. I'm basically sticking my finger in the wind. In many real world applications, such as robotics, the rewards may be very sparse requiring dense reward shaping, such that the models end up "hacking" the metrics being optimized ala Goodhart's Law:

> When a measure becomes a target, it ceases to be a good measure.

Self-play introduces challenges of its own. Take AlphaGo and AlphaZero which are trained by playing against themselves basically. The goal of such models is to win games against an opponent. Duh, right? Elo rating is a metric that rates human players in such competitive two-player games, but it can also be used for an AI vs human or AI (ie itself or worse versions of itself). But what does the Elo rating really tell us if our AI is just beating bad humans or bad versions of itself? Elo in itself isn't an objective metric. The same is true for BasketWorld in a large sense. It's not easy for me to simply look at "reward" (which of course can just be points scored or allowed) because the context of the competition matters. We're not just trying to build an agent that simply "wins" but one that actually learns the game and evolves strategies to beat better and better competition. Based on our everyday experience as basketball fans we know there is a difference between an NCAA game and an NBA game even if the final scores may be relatively similar. The strategies utilized in-game can be very different depending personnel, coaching philosophy, and (maybe especially) rules. 

### BasketWorld: The Game with the Game

We do, of course, have access to a myriad of different metrics that are typically monitored during RL training, for example, reward per episode, KL divergence, episode length, etc. Below are a few example charts during a typical training session.

![Episode Length](../images/Offense%20rollout_ep_len_mean.png)

![Episode Reward](../images/Offense%20rollout_ep_rew_mean.png)

![KL Divergence](../images/Offense%20train_approx_kl.png)

![Training Loss](../images/Offense%20train_loss.png)

Metrics such as these have their use for monitoring but they are not at all interpretable in the sense that AUC or RMSE is for their counterparts. I can't look at a KL Divergence or Episode Reward chart and tell you if my agents learned how to execute a pick and roll (which has basically become my holy grail of outcomes). 

So what can I use? Well, not surprisingly, I decided to "go to the tape". I'm focusing heavily on building a suite of visualization tools to simulate games, understand how my BW agents are "thinking", and to help guide the addition of new features, including rules and physical contraints. Critically, during training of a model I periodically output evaluation animations to see how agents evolve. While that helps I wanted to go even deeper down the rabbit hole and make my own interactive "video game" so I could play against the trained AI agent or watch and record what they do against each other in real time. The game (which is located in the [BW github repo](https://github.com/EvanZ/basketworld) under `/app`) enables me to select a run id from MLflow as well as any of the offensive and defensive policies created during training.

![Welcome Screen](../images/welcome%20screen.png)

When the game is started it presents a court with the agents and a set of player controls for the side I chose to play.

![Game Startup](../images/initial_game_play_screen.png)

The example above is for a 2-on-2 trained set of agents (there is a separate policy for offense and defense). Here I'm playing as offense (blue) and the defense is being controlled by the AI defensive policy. Game play is turn-based. I select the actions for each of the players on offense. The current ball handler is highlighted by a dashed orange circle. The currently selected player is highlighted by a yellow circle. In the screen above Player 0 is the currently selected player. On the right of the screen are the "actions" available for that player. They are all the Move actions, because the player without the ball can only choose to move in one of the 6 hexagonal directions or not to move (if no movement action is selected explicitly). 

Below is what the screen looks like when I select the ball handler (Player 1):

![Player 1 Selected](../images/initial_game_play_player_1.png)

The ball handler has more possible actions, the 6 movement actions plus 6 additional passing actions ("send" icon) and a shooting action (the button in the middle). Once I have selected an action for each player, I click "Submit Turn" and then the screen updates immediately with my action and the actions chosen by the AI. Here I've selected the Move NW action (up and to the left) for both Players 0 & 1:

![First Move](../images/first_turn.png)

You can see that one "second" has come off the shot clock. In the current iteration of the model each step of game play corresponds to one second of "real time", but obviously future iterations could make steps at the sub-second level. In this screenshot you can also see that on defense Player 3 who is "defending" the ball handler has also moved in the same direction, which I interpret as "staying in front". This is a learned behavior clearly, as earlier in the training process agents move randomly much of the time. 

![Player 1 Zoom](../images/player_1_zoom.png)

Now if we zoom in on Player 1 (above), we can see a bunch of numbers. Those numbers reveal the probability of each action according to the policy learned by the AI. In other words, if the AI was playing on its own, it would select an action according to this distribution. The number on top of each pair corresponds to the Move action, while the number underneath is the Pass move in the corresponding direction. The bold number overlaid on the ball handler is the probability of selecting the Shoot action. So, for example, in the screenshot above the AI would almost certainly try to move directly to the West (ie left) towards the basket, as the probability of Move West is > 95%. As the human player I can ignore that probability and take whatever action I want.

In the player controls section you can see actions are colored on a scale from blue to orange. And there are numeric labels on each action button. Those colors and numbers correspond to the value network output from the agent. Not to get too deep into the weeds, because I still don't understand all the weeds to be honest, but those values are based on what a separate model called the "critic" thinks of the "actor" model selecting the actions. In this case you can see that the more valuable actions are all the movement actions towards the basket. This is not surprising as you will shortly see. Here are a few more frames from this game.

![19 seconds](../images/game_play_manual_19s.png)

![13 seconds](../images/game_play_manual_13s.png)

![11 seconds](../images/game_play_manual_11s.png)

Notice the shooting action probability is now 99%. Now just to be clear that doesn't mean that the probability of making the shot is 99%. In fact *that* probability (the probability of a shot attempt being successful) is shows in the central button of the Player 1 controls on the right side of the screen and is 90% from a distance of 1 hexagon from the basket. So let's attempt a shot and see what happens!

![Turnover!](../images/game_play_manual_shot_attempt.png)

Oh shit, that's embarrasing. Our ball handler turned the ball over. Well, it happens. Here's the full episode (note that the court is flipped upside down in this animation):

![manual game play episode](./../app/backend/episodes/episode_20250803_123129.gif)

I have programmed in the concept of "defensive pressure", which currently is defined that if there is a defender within one hexagon unit of the ball handler there is a 5% chance of a turnover. That means, for example, if a defender can stay tied to the ball handler over the course of the possession there is a pretty good chance of a turnover if the ball handler doesn't shoot or pass first. In fact we can see the defense has little incentive to defend the player off the ball and his defender tends to come over and "help". Even still, so far I rarely see passing. It's usually feast or famine with the ball handler trying to drive to the basket and shoot from very close range. This is what the agents have learned based on the current rules and constraints of the environment I've built. 

Here's an example episode in full "AI mode" where I let both sides be controlled by the learned policies. It's a made shot in this case. It's neat to the see the player trajectories this way because you see immediately that it's not at all random. There's a clear persistence to the movement that was learned by the AI (on both sides of the ball).

![made shot full AI](../images/ai_mode_made_shot.png)

![movie version](../app/backend/episodes/episode_20250803_123751.gif)

### Discussion

So obviously there's a lot more to do to make a realistic version of basketball here, but at least, we're building the tool set to iterate faster and "eye test" what is being built in the process. It's really helping me understand the complexity of the problem space. One other thing I really like about this approach is that it gives me ideas for future analytics projects. For example, how cool would it be to annotate actual basketball games with the probabilities and value functions we learn from our AI policies? Very cool, indeed. Such tools would provide so much insight into player (and coach) decision-making and counterfactual reasoning and evaluation of games. Just a thought. 

### Technical Details

For those interested in the technical details, BasketWorld is built on the Gymnasium standard, with agents trained using the PPO algorithm from the Stable-Baselines3 (https://stable-baselines3.readthedocs.io/en/master/#) library and experiment tracking managed by MLflow. The project is open source on github under a MIT license. If you like the project please support it by giving it a star. And of course, please leave your comments or questions in the comments section. I'd love to get your feedback so I can learn too!

https://github.com/EvanZ/basketworld