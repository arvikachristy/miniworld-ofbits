#!/usr/bin/python
import os
import numpy as np

class Output:
    'export the reward'
    #use f = Output() to initiate, use f.saveEpisode(int_episode_number, int_reward) to write a record in both files

    def __init__(self, summaryFreq=20, outputDirName="evaluation", rewardFileName="rewards", summaryFileName="summary"):
        self.summaryFreq = summaryFreq
        if outputDirName != "":
            #Make a path for our model to be saved in.
            if not os.path.exists(outputDirName):
                os.makedirs(outputDirName)
            rewardFileName = os.path.join(outputDirName, rewardFileName)
            summaryFileName = os.path.join(outputDirName, summaryFileName)
        nextAvailableFileNumber = self.getNextAvailableFileNumber(rewardFileName, summaryFileName) # without .txt extension
        self.rewardFileName = rewardFileName + nextAvailableFileNumber + ".txt"
        self.summaryFileName = summaryFileName + nextAvailableFileNumber + ".txt"
        self.rewards = []

    # Return the smallest available integer (None, 2, 3,...) which, when appended to reward and summary filenames,
    # avoids name clashes with existing files.
    def getNextAvailableFileNumber(self, rewardFileName, summaryFileName):
        nextNumber = 1
        # default: no number
        if os.path.isfile(rewardFileName + ".txt"):
            nextNumber += 1
            while os.path.isfile(rewardFileName + "-" + str(nextNumber) + ".txt"):
                nextNumber += 1
        # neither file exists: no number is appended
        elif not os.path.isfile(summaryFileName + ".txt"):
            return ""
        # summaryFile exists, rewardFile doesn't
        else:
            nextNumber += 1
        # summaryFile exists, rewardFile may exist
        while os.path.isfile(summaryFileName + "-" + str(nextNumber)):
            nextNumber += 1
        return "-" + str(nextNumber)

    def open(self):
        self.rewardFile = open(self.rewardFileName, "a")
        self.summaryFile = open(self.summaryFileName, "a")

    def close(self):
        self.rewardFile.close()
        self.summaryFile.close()

    def addReward(self, episode, reward):
        self.rewards.append(reward)
        self.rewardFile.write(str(episode) + "\t" + str(reward) + "\n")

    def addSummary(self, episode):
        self.summaryFile.write(
            str(episode) + "\t" +
                str(np.mean(self.rewards[-10:])) + "\t" +
                str(np.mean(self.rewards[-100:])) + "\t" +
                str(np.mean(self.rewards[-1000:])) + "\n")

    def saveEpisode(self,episode,reward):
        self.open()
        self.addReward(episode, reward)
        if episode % self.summaryFreq == 0:
            self.addSummary(episode)
        self.close()


