
# metagraph.py -- Tensorflow save node information and meta data used for 
# computing with MetaGraph  structure .
#
# MetaGraphDef data structure defined in MetaGraphDef Protocol Buffer .
#
# MetaGraphDef information saved as a file with suffix '.ckpt.meta' in 
# model path in binary format .
#
# Tensorflow provides export_meta_graph() to export MetaGraphDef Protocol Buffer
# into a json file .

'''
message MetaGraphDef{
	MetaInfoDef meta_info_def = 1;

	GraphDef graph_def = 2;
	SaverDef saver_def = 3;
	map<string, CollectionDef> collection_def = 4;
	map<string, SignatureDef> signatrue_def = 5;
}
'''
import tensorflow as tf

'''
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")

result = v1 + v2

saver = tf.train.Saver()
# Export tensorflow MetaGraphDef into json .
saver.export_meta_graph("./model/model.ckpt.meta.json", as_text=True)
'''

'''
// Attribute meta_info_def defines tensorflow meta data and
// computation information .
message MetaInfoDef{
	string meta_graph_version = 1;
	// Record all computaion information on tensorflow graph
	OpList stripped_op_list = 2;			
	google.protobuf.Any any_info = 3;
	repeated string tags = 4;
}
// OpList is the list of OpDef structure .
message OpDef{
	string name = 1;			// Computation name .
	repeated ArgDef input_arg = 2;		// Input arguments list .
	repeated ArgDef output_arg = 3;		// Output arguments list .
	reapeatd AttrDef attr = 4;		// Other computation arguments list .

	string summary = 5;
	string description = 6;
	OpDeprecation deprecation = 8;

	bool is_commutative = 18;
	bool is_aggregate = 16;
	bool is_stateful = 17;
	bool allows_uninitialized_input = 19;
};
// Attribute graph_def focus on link-structure of computaion , 
// and records the node information on graph .
// graph_def defined by GraphDefProtocol Buffer .
message GraphDef{
	repeated NodeDef node = 1;		// Record GraphDef information .
	VersionDef versions = 4;
};

message NodeDef{
	string name = 1;			// Unique ID for node .
	string op = 2;				// Computating operation name .
	repeated string input = 3;		// Computaion inputs .
	string device = 4;			// CPU or GPU .
	map<string, AttrValue> attr = 5;	// Cofig information .
};
// Attribute saver_def records arguments used for model persistence .
message SaverDef{
	string filename_tensor_name = 1;		// Tensor name for saving filename to persistent model .
	string save_tensor_name = 2;			// Node name to persistent model .
	string restore_op_name = 3;			// Node name to restore model .
	// tensorflow.train.Saver class to manage history model .
	int32 max_to_keep = 4;				// Limit the number to maintain models . 
	bool shared = 5;
	float keep_checkpoint_every_n_hours = 6;	// Increment number of maintaining models every n hours .

	enum CheckpointFormatVersion{
		LEGACY = 0;
		V1 = 1;
		V2 = 2;
	}
	CheckpointFormatVersion version = 7;
}
// Attribute collection_def maintain different collections in tensorflow Graph .
// collection_def maps collection name to collection contents .
message CollectionDef{
	message NodeList{
		repeated string value = 1;	
	}		

	message BytesList{
		repeated bytes value = 1;	
	}

	message Int64List{
		repeated int64 value = 1 [packed = true];	
	} 

	message FloatList{
		repeated float value = 1 [packed = true];	
	}

	message AnyList{
		repeated google.protobuf.Any value = 1;	
	}

	oneof kind{
		NodeList node_list = 1;
		BytesList bytes_list = 2;
		Int64List int64_list = 3;
		FloatList float_list = 4;
		AnyList any_list = 5;
	}
}
// Attribute checkpoint produced by tensorflow.train.Saver class
// maintaining all models persistented by tf.train.Saver .
message CheckpointState{
	string model_checkpoint_path = 1;
	repeated string all_model_checkpoint_paths = 2;
}

'''

# tensorflow.train.NewCheckpointReader get all variables in checkpoint .
reader = tf.train.NewCheckpointReader('./model/model.ckpt')

# Get all variables list to be a dictionary .
global_variables = reader.get_variable_to_shape_map()
for variable_name in global_variables:
	# variable_name as variable name , global_variable[variable_name] as the shape
	print(variable_name, global_variables[variable_name])

# Get variable v1 value .
# print("Value for variable v1 is ", reader.get_tensor("v1"))



















































