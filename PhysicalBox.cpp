/*
 * PhysicalBox.cpp
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#include <iostream>
#include "PhysicalBox.h"

void BoxParameters::inspect() const
{
	std::cout << " This is a box " << std::endl;
	std::cout << " dX : " << dX << std::endl;
	std::cout << " dY : " << dY << std::endl;
	std::cout << " dZ : " << dZ << std::endl;
}

