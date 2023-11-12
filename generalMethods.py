#//**************************************************************
#// generalMethods_RQM.py
#//
#// Purpose: This file contains a punch of general use methods for python.
#//
#// R0: 07-Mar-2023 Ro Muhoberac - Start Date
#// R1: 13-Mar-2023 Ro Muhoberac - Added Comments
#// R2: 14-Aug-2023 Ro Muhoberac - Updated Methods + Documentation
#// 
#//**************************************************************

import subprocess
import sys
import os
import sys
import csv

# Class file Interaction Methods (fIM) contains methods that interact with files 
class fIM:
      
   # readFile (rF) "reads" a given file into a list
   def rF(fileName):
      tempFile = open(fileName, "r+")
      temp = tempFile.read().split("\n")
      tempFile.close()
      return temp
   
   # writeFile (wF) "writes" a given set of contents to a given file
   def wF(fileName, fileInfo):
      file = open(fileName, "w+")
      for line in fileInfo:
         file.write(line + "")
      file.close()
      return True                   # update method to use writelines() method?
   
   # Method CSVReader (csvR) reads a given CSV and optionally returns all the contents in said file.
   def csvR(readFile):
      with open(readFile, 'r') as file:
        csvreader = csv.reader(file)
        csvList = []
        for row in csvreader:
          csvList.append(row)
        return csvList
   
   # Method CSVWriterAdd (csvWA) adds rows to a given CSV file.
   def csvWA(readFile, input):
      with open(readFile, 'a',newline='') as file:
         csvwriter = csv.writer(file)
         csvwriter.writerow(input)
         
   # Method CSVWriterReplace (csvWR) replaces a given CSV file's contents with user input
   def csvWR(readFile, input):
      with open(readFile, 'w',newline='') as file:
               csvwriter = csv.writer(file)
               csvwriter.writerow(input)
               
# Class string Interaction Methods (strIM) contains methods that interact/change strings.
class strIM: 
   # splitByCharacter (sBC) returns a list of each character in a string
   def sBC(input):
      return list(input) 
     
   # getCharacterID (gCID) returns a numerical representation of a string (WIP)
   def gCID(playerName):
      characterCharList = strIM.sBC(playerName)
      print(characterCharList)
      
   # getAlphabetUpperCase (gAUC) returns a list of the alphabet in uppercase   
   def gAUC():
      upperAlphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
      return upperAlphabet
      
   # getAlphabetLowerCase (gALC) returns a list of the alphabet in lowercase
   def gALC():
      lowerAlphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
      return lowerAlphabet
      
   # getAlphabetNumericPairs (gANP) returns a list of integers to represent 1-26 Alphabet Characters (A - 1, B - 2, etc..)
   def gANP():
      numPairs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
      return numPairs
   
   
class sysIM:
   # clearScreen (cS) "Clears" the Screen using newline characters  
   def cS():
      os.system('cls')
   
   # startProgram (sP) starts a program.
   def sP(program, exit_code=0):
      subprocess.Popen(program)
      sys.exit(exit_code)   