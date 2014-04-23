
### Change this to include the shape-specific definitions
require './tube'


### No changes should be required beyond this point

SHAPE = Shape.upcase()

Member_definitions = []
Params.each { |x|
	Member_definitions.push("#{x[0]} #{x[1]}_")
}

Constructor_arguments = []
Params.each { |x|
	Constructor_arguments.push("const #{x[0]} &#{x[1]}" )
}

Initialization_list = []
Params.each { |x|
	Initialization_list.push("#{x[1]}_(#{x[1]})")
}

Copy_constructor = []
Params.each { |x|
	Copy_constructor.push("#{x[1]}_(other.#{x[1]}_)")
}

Print_arguments = []
Params.each { |x|
	Print_arguments.push("#{x[1]}_")
}
