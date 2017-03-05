#!/usr/bin/python
class output:

   'export the reward'
   #use f= output() to initate,  use f.newEandR(int_episode_number ,int_reward) to write a record in both 2files
   

   def __init__(self):
      #self.file = open("reward.txt", "a")
      self.rewardlist=[]


   def add(self,reward,average):
      self.file1.write(reward+"\n");
      self.file2.write(average+"\n");
      print "add " 
 
   def array2str(self,array):
      t =','.join(array)
      print t 

   def int2str(self,array):
      t =','.join(str(l) for l in array)
      print t 

   def newEandR(self,episode,reward):
      self.open()

      E_R = str(episode)+','+str(reward)
      E_R = str(E_R)

      self.rewardlist.append(reward)


      N100 = self.getaverage(100)
      N10 = self.getaverage(10)
      average = N10 +"," +N100
      average = E_R+","+average
      

      self.add(E_R,average)    

      self.close()
      print "add e and r "

   def newR(self,e,r):
      str1 = [e,r]
      str1 = str(str1)
      self.add(str1)
      print "add e and r "


   def close(self):
      self.file1.close()
      self.file2.close()
      #print "close!"

   def open(self):
      self.file1 = open("reward.txt", "a")
      self.file2 = open("average.txt", "a")

   def getaverage(self,number):
      if len(self.rewardlist)>=number:
          newlist =self.rewardlist[-number:]
          aver = sum(newlist)/number
          return str(aver)
      else :
          return 'N/A'

    
#f = output()
#f.add()
#f.close()

