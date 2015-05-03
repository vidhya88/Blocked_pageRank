package LSIProject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;


public class BlockPR {
	
	public static final long totalNodes = 6;//685230;//6;//685230;	
	public static final int totalBlocks = 2;//68;// 2;//68;	
	public static final int precision = 10000;	
	private static final Float thresholdvalue = 0.001f;	
	
	public static enum Hadoop_Counter{
	    RESIDUAL_ERROR
	};
	
	public static class Block_Mapper extends Mapper<LongWritable, Text, Text, Text> 
	{
		public static int getBlockID(int nodeID) {
			int avg_partitionSize = 3;//10000;//3;//10000;
			//taken from block.txt
			int[] boundary_nodes = {0,3};/*{ 0, 10328, 20373, 30629, 40645,
					50462, 60841, 70591, 80118, 90497, 100501, 110567, 120945,
					130999, 140574, 150953, 161332, 171154, 181514, 191625, 202004,
					212383, 222762, 232593, 242878, 252938, 263149, 273210, 283473,
					293255, 303043, 313370, 323522, 333883, 343663, 353645, 363929,
					374236, 384554, 394929, 404712, 414617, 424747, 434707, 444489,
					454285, 464398, 474196, 484050, 493968, 503752, 514131, 524510,
					534709, 545088, 555467, 565846, 576225, 586604, 596585, 606367,
					616148, 626448, 636240, 646022, 655804, 665666, 675448, 685230 };*/
			int blockID = (int) Math.floor(nodeID / avg_partitionSize);
			if (nodeID < boundary_nodes[blockID]) 
				return blockID--;
			return blockID;
		}
		
		protected void map(LongWritable in_key, Text in_value, Context context)
				throws IOException, InterruptedException {		
			String in_line = (in_value.toString()).trim();	
			String[] temp = in_line.split("\\s+");
			// node + pagerank +degree + edgelist			
			String edgeList = temp.length == 4?temp[3]:"";	
			//get block id
			Integer blockID = new Integer(getBlockID(new Integer (temp[0])));
			//Emit key is block id to grp by blocks and value is BLOCKPR nodeid degree edgelist
			Text key = new Text(blockID.toString());
			Text value = new Text("BLOCKPR " + temp[0] + " " + temp[1] + " "+ edgeList);
			System.out.println(key + " "+ value );
			context.write(key, value);
			//check if edges are der from the block and emit EDGE or BOUNDARY based on whether the edge is inbound or outbound
			if (edgeList != "") {
				String[] edgeListArray = edgeList.split(",");
				for (int i = 0; i < edgeListArray.length; i++) {
					Integer blockIDOut = new Integer(getBlockID(Integer.parseInt(edgeListArray[i])));
					key = new Text(blockIDOut.toString());			
					if (blockIDOut.equals(blockID)) {				
						value = new Text("EDGE " + temp[0] + " " + edgeListArray[i]);				
					} else {
						Float pageRankFactor = new Float(new Float(temp[1]) / new Integer(temp[2]));					
						String pageRankFactorString = String.valueOf(pageRankFactor);
						value = new Text("BOUNDARY " + temp[0] + " " + edgeListArray[i] + " " + pageRankFactorString);
					}
					//emit key and value from mapper
					context.write(key, value);
					System.out.println(key + " "+ value );
				}
			}
		}
		
		
	}
	
	
	public static class Block_Reducer extends Reducer<Text, Text, Text, Text> {
		
		public class Node {
			public String nodeID = "";
			public String edgeList = "";
			public double pageRank = 0.0f;
			public Integer degrees = 0;
			
		}
		//global map to hold PR values for each node
		public HashMap<String, Double> PRmap = new HashMap<String, Double>();	
		public HashMap<String, ArrayList<String>> BE = new HashMap<String, ArrayList<String>>();
		public HashMap<String, Double> BC = new HashMap<String, Double>();
		public HashMap<String, Node> nodeDataMap = new HashMap<String, Node>();
		public ArrayList<String> vlist = new ArrayList<String>();
		public double d = (Double) 0.85;
		public int maxIterations = 6;
		public double threshold = 0.001f;
		
		
	/*
	 * void IterateBlockOnce(B) {
	    for each ( v ∈ B ) { for
	     NPR[v]=0
	        for( u where <u, v> ∈ BE ) {
	            NPR[v] += PR[u] / deg(u);
	        }
	        for( u, R where <u,v,R> ∈ BC ) {
	            NPR[v] += R;
	        }
	        NPR[v] = d*NPR[v] + (1-d)/N;
	    }
	    for( v ∈ B ) { PR[v] = NPR[v]; }
	 */
		
		protected Double IterateBlockOnce() {		
			ArrayList<String> uList = new ArrayList<String>();
			double NPR = 0.0f;
			double R = 0.0f;
			double error = 0.0f;	
			for (String v : vlist) {
				NPR = 0.0f;
				double prevPR = PRmap.get(v);
				//check if v is in BE or BC and adjust rank
				if (BE.containsKey(v)) {
					uList = BE.get(v);
					for (String u : uList) {
						Node uNode = nodeDataMap.get(u);
						NPR += (PRmap.get(u) / uNode.degrees);
					}
				}			
				if (BC.containsKey(v)) {
					//add boundary nodes
					R = BC.get(v);
					NPR += R;
				}	
		        //calculate NPR
				NPR = (d * NPR) + ((1 - d) / BlockPR.totalNodes);
				// update the map
				PRmap.put(v, NPR);
				//calculate error
				error += Math.abs(prevPR - NPR) / NPR;
			}
			error = error / vlist.size();
			return error;
		}
		
		@Override
		protected void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {		
			Text input = new Text();
			String[] inputTokens = null;		
			// initialize/reset all variables
			Double pageRankOld = (Double) 0.0;
			Double residualError = (Double) 0.0;		
			String output = "";
			Integer maxNode = 0;		
			ArrayList<String> temp = new ArrayList<String>();
			double tempBC = 0.0f;
			vlist.clear();
			PRmap.clear();
			BE.clear();
			BC.clear();
			nodeDataMap.clear();			
			while (values.iterator().hasNext()) {
				input = values.iterator().next();
				inputTokens = input.toString().split(" ");			
				// if first element is PR, it is the node ID, previous pagerank and outgoing edgelist for this node
				if (inputTokens[0].equals("BLOCKPR")) {
					String nodeID = inputTokens[1];
					pageRankOld = Double.parseDouble(inputTokens[2]);
					PRmap.put(nodeID, pageRankOld);
					Node node = new Node();
					node.nodeID=nodeID;
					node.pageRank=pageRankOld;
					if (inputTokens.length == 4) {
						node.edgeList= inputTokens[3];
						node.degrees=inputTokens[3].split(",").length;
					}
					vlist.add(nodeID);
					nodeDataMap.put(nodeID, node);
					// keep track of the max nodeID for this block
					if (Integer.parseInt(nodeID) > maxNode) {
						maxNode = Integer.parseInt(nodeID);
					}
					
				// if BE, it is an in-block edge
				} else if (inputTokens[0].equals("EDGE")) {			
					
					if (BE.containsKey(inputTokens[2])) {
						//Initialize BC for this v
						temp = BE.get(inputTokens[2]);
					} else {
						temp = new ArrayList<String>();
					}
					temp.add(inputTokens[1]);
					BE.put(inputTokens[2], temp);
					
				// if BC, it is an incoming node from outside of the block
				} else if (inputTokens[0].equals("BOUNDARY")) {
					if (BC.containsKey(inputTokens[2])) {
						//Initialize BC for this v
						tempBC = BC.get(inputTokens[2]);
					} else {
						tempBC = 0.0f;
					}
					tempBC += Double.parseDouble(inputTokens[3]);
					BC.put(inputTokens[2], tempBC);
				}		
			}
			
			int i = 0;
			do {
				i++;
				residualError = IterateBlockOnce();
				//System.out.println("Block " + key + " pass " + i + " resError:" + residualError);
			} while (i < maxIterations && residualError > threshold);

					
			// compute the ultimate residual error for each node in this block
			residualError = 0.0;
			for (String v : vlist) {
				Node node = nodeDataMap.get(v);
				residualError += Math.abs(node.pageRank - PRmap.get(v)) / PRmap.get(v);
			}
			residualError = residualError / vlist.size();
			//System.out.println("Block " + key + " overall resError for iteration: " + residualError);
			
			// add the residual error to the counter that is tracking the overall sum (must be expressed as a long value)
			long residualAsLong = (long) Math.floor(residualError * BlockPR.precision);
			context.getCounter(BlockPR.Hadoop_Counter.RESIDUAL_ERROR).increment(residualAsLong);
			
			// output should be 
			//	key:nodeID (for this node)
			//	value:<pageRankNew> <degrees> <comma-separated outgoing edgeList>
			for (String v : vlist) {
				Node node = nodeDataMap.get(v);
				output = PRmap.get(v) + " " + node.degrees + " " + node.edgeList;
				Text outputText = new Text(output);
				Text outputKey = new Text(v);
				context.write(outputKey, outputText);
				//if (v.equals(maxNode.toString())) {
					System.out.println("Block:" + key + " node:" + v + " pageRank:" + PRmap.get(v));
				//}
			}
				
			cleanup(context);
		}

	}
	
	public static void main(String[] args) throws Exception {

		String input_graph = args[0];
		String output = args[1];		
		Float residualError = 10.0f;
		
     for(int i=0;i<6 && residualError > thresholdvalue;i++ )
     {
    	 //set job config
        	Job job = new Job();
            job.setJobName("Block_PageRank"+ (i+1));          
            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            job.setJarByClass(BlockPR.class);
            job.setMapperClass(Block_Mapper.class);
            job.setReducerClass(Block_Reducer.class);                            
            FileInputFormat.addInputPath(job, i==0?(new Path(input_graph)):new Path(output + "/iteration"+(i)));
            FileOutputFormat.setOutputPath(job, new Path(output + "/iteration"+(i+1)));
            //start job
            job.waitForCompletion(true);
            //calculate error
            residualError = (float) job.getCounters().findCounter(Hadoop_Counter.RESIDUAL_ERROR).getValue() / precision  / totalBlocks;            
            System.out.println("Residual error"  + residualError.toString());
            job.getCounters().findCounter(Hadoop_Counter.RESIDUAL_ERROR).setValue(0L);
        } 
        
    }
}


