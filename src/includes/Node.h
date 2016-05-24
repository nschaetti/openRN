/*
 * Node.h
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
 */

#ifndef __NODE_H__
#define __NODE_H__

#include <string>
#include <list>

/*! \namespace openRN
 *
 * Global namespace for the openRN library
 */
namespace openRN
{
		
	/*! \class Node
	* \brief Basic node class
	*
	* This is the basic class for a node
	*/ 
	class Node
	{
	/**** PUBLIC ****/
	public:	
	
		/*!
		 * \brief Constructor
		 *
		 * Node's constructor
		 */
		Node(void);
	
	/**** PRIVATE ****/
	private:
	
		/*!
		 * \brief Data transformations done just after receiving inputs
		 *
		 * This function compute data just after receiving inputs
		 */
		virtual void preProcessing(void);
		
		/*!
		 * \brief Data transformations done just before sending outputs
		 *
		 * This function compute data just before sending outputs
		 */
		virtual void postProcessing(void);
	
	};
	
};

#endif