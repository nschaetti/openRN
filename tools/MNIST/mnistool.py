#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 * mnistool.py
 * 
 * Copyright 2015 Nils Schaetti <n.schaetti@gmail.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 """

# import
import argparse
import cPickle
from mnist import *
import sys
import time

################################################################
# FUNCTIONS
################################################################

"""
* Enregistre une image
* target_file File to save the image
* image image array 
"""
def saveImage(target_file, image):
	print u'\r[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Saving image " + target_file,
	
	# Nouveau fichier
	mon_fichier = open(target_file,"w")
	
	# Pour chaque lignes
	for j in np.arange(image.shape[1]):
		# Pour chaque colonnes
		for i in np.arange(image.shape[0]):
			mon_fichier.write("   " + str(image[i,j]))
		mon_fichier.write("\n")
	
	# Ferme le fichier
	mon_fichier.close()
#end saveImage

"""	
* Enregistre le fichier de correspondances
* target_file File where to save the labels
* labels Labels array
"""
def saveLabels(target_file, labels):
	print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Saving corr file " + target_file
	
	# Nouveau fichier
	mon_fichier = open(target_file,"w")
	
	# Pour chaque labels
	for i in np.arange(0, labels.shape[0]):
		mon_fichier.write("   " + str(labels[i]) + "\n")
	
	# Ferme le fichier
	mon_fichier.close()
#end saveLabels

"""
* Enregistre la description
* target_dir Where to save images
* image_size
"""
def saveDesc(target_dir, image_size, rep):
	
	# Nouveau fichier 
	mon_fichier = open(target_dir+"/desc","w")
	
	# Ecrits
	mon_fichier.write(str(image_size) + "\n")
	mon_fichier.write(str(rep))
	
	# Ferme le fichier
	mon_fichier.close()
#end saveDesc

def exportMatlab(digitImport, target_dir, image_size, rep, nb_train, nb_test):
	
	# Create directories
	print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u'] Creating directories...'
	os.makedirs(target_dir + "/images")
	os.makedirs(target_dir + "/test")
	
	# Save description
	print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Saving description..."
	saveDesc(target_dir, image_size, 4)
	
	# For each training images
	print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Saving training images..."
	for i in np.arange(0, nb_train):
		saveImage(target_dir + "/images/image_" + str(i+1) + ".dat", digitImport.trainTSImages[i*image_size:i*image_size+image_size,:])
	print ""
	
	# For each test images
	print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Saving test images..."
	for i in np.arange(0, nb_test):
		saveImage(target_dir + "/test/image_" + str(i+1) + ".dat", digitImport.testTSImages[i*image_size:i*image_size+image_size,:])
	print ""
	
	# Saving labels
	print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Saving labels..."
	saveLabels(target_dir + "/train_labels", digitImport.trainLabels[0:nb_train])
	saveLabels(target_dir + "/test_labels", digitImport.testLabels[0:nb_test])
	
#end exportMatlab

################################################################
# MAIN
################################################################

# Main
if __name__ == "__main__":
	
	# Parse
	parser = argparse.ArgumentParser(description="MNIST data tool generator")
	parser.add_argument('--input', metavar='i', type=str, required=True, help="Path to Python object file containing original MNIST database")
	parser.add_argument('--output', metavar='o', type=str, required=True, help="Path to output file or directory")
	parser.add_argument('--format', metavar='f', type=str, default="python", help="output format (python,matlab)")
	parser.add_argument('--start', metavar='s', type=int, help="Starting position")
	parser.add_argument('--end', metavar='e', type=int, help="Ending position")
	parser.add_argument('--rotation', metavar='r', type=str, required=True, help="List of angles (ex 0,30,60,60)")
	parser.add_argument('--size', metavar='m', type=int, default=28, help="Output images size")
	parser.add_argument('--resize', metavar='z', type=str, default="none", help="Resize before/after rotation (none,before,after,both)")
	parser.add_argument('--train', metavar='t', type=int, default=60000, help="Training set size")
	parser.add_argument('--test', metavar='b', type=int, default=10000, help="Test set size")
	args = parser.parse_args()
	
	try:
		# Try to open MNIST
		print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Loading MNIST dataset from " + args.input
		digitImport = MNISTImporter()
		digitImport.Load(args.input)
		
		# Get transforms
		rep = 0
		angles = args.rotation.split(",")
		
		# Apply each transforms
		first = True
		for angle in angles:
			print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Applying transformatio angle " + angle
			
			# Resize if first time or resize before
			if first or args.resize == "before" or args.resize == "both":
				digitImport.resizeImages(size = args.size)
				first = False
			#end if
			
			# Rotation
			if int(angle) != 0:
				digitImport.rotateImages(angle = int(angle))
			#end if
			
			# Resize if after is set
			if args.resize == "after" or args.resize == "both":
				digitImport.resizeImages(size = args.size)
			#end if
			
			# Add timeserie
			digitImport.addTimeserie()
			rep += 1
		#end for
		
		# Generate labels
		print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Generating labels..."
		digitImport.generateLabels()
		
		# Export
		if args.format == "matlab":
			exportMatlab(digitImport, args.output, args.size, rep, args.train, args.test)
		else:
			print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u"] Saving dataset to " + args.output
			digitImport.Save(args.output)
	except:
		e = sys.exc_info()
		print u'[' + unicode(time.strftime("%Y-%m-%d %H:%M")) + u'] \033[91m' + str(e) + '\033[0m'
	
#end Main
