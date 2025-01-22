#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kdtree.h" 

#define INITIAL_SIZE	5000
#define INCREASEMENT	500
#define UNDEFINE	50.0
 
typedef struct Core_Object{
	int objectID;
	int *directlyDensityReachable_ID;
	int size_DDR;
	int capacity;
	float reachabilityDistance;
	float coreDistance;
	int processed;
} Core_Object;
 
Core_Object *core_object_set;
Core_Object *object_neighbor;
 
typedef struct QueueNode{
	int data;
	struct QueueNode *next;
	float reachabilityDistance;
} QueueNode;

typedef struct LinkQueue{
	QueueNode *front;
	QueueNode *rear;
} LinkQueue;
 
typedef struct Cluster_Order_Node{
	int objectID;
	float core_distance;
	float reachability_distance;
} Cluster_Order_Node;

Cluster_Order_Node *cluster_ordering;
 
float epsilon;
int MinPts;
char filename[200];
int nspots;
int ndim;
int core_object_size;
int cluster_order_counter = 1;
 
float *dataBase;
float *distanceMatrix;
 
void optics_init();
void readDataFromFile();
float calculateDistance_betweenTwoObject(int, int);
void calculateDistanceMatrix();
void createCoreOjectSet();
void createObjectNeighborSet();
void OPTICS();
void setCoreDistance(int);
int isProcessed(int);
void saveToClusterOrdering(int, int);
void update(int);
void ExpandClusterOrder(int);
void writeClusterOrdering2File();
 
//queue
void initialQueue(LinkQueue*);
void insertQueue(LinkQueue*, int, float);
void deleteQueue(LinkQueue*, int*);
int isQueueEmpty(LinkQueue);
void printQueue(LinkQueue);
void decrease(LinkQueue*, int, float);
void testQueue();
 
LinkQueue orderSeeds;
//queue
 
int main(int argc, char* argv[])
{
	if( argc != 6 )
	{
		printf("This algorithm require four parameters"
				"\n\tthe neighborhood distance"
				"\n\tthe MinPts"
				"\n\tthe filename contains data"
				"\n\tthe size of data"
				"\n\tthe dimension of data"
				"\n");
		exit(0);
	}
 
	epsilon = atof(argv[1]);
	MinPts = atoi(argv[2]);
	strcat(filename, argv[3]);
	nspots = atoi(argv[4]);
	ndim = atoi(argv[5]);
 
	optics_init();
	readDataFromFile();
	calculateDistanceMatrix();
	createCoreOjectSet();
	createObjectNeighborSet();
 
	//testQueue();
 
	OPTICS();
	writeClusterOrdering2File();
	return 0;
}



/*
 * do some initialization work:
 * 	create the dynamic array @dataBase which be used to store the origin data
 * */
void optics_init()
{
	// optics_init dataBase
	dataBase = (float*)malloc(ndim * (nspots + 1) * sizeof(float));
	if( !dataBase )
	{
		printf("optics_init: dataBase malloc error: 0\n");
		exit(0);
	}
	
	// optics_init distance matrix 
	distanceMatrix = (float*)malloc((nspots + 1) * (nspots + 1) * sizeof(float));
	if( !distanceMatrix )
	{
		printf("optics_init: distance matrix malloc error: 0");
		exit(0);
	}
 
	// optics_init cluster_ordering
	cluster_ordering = (Cluster_Order_Node*)malloc((nspots + 1) * sizeof(Cluster_Order_Node));
	if( !cluster_ordering )
	{
		printf("optics_init: cluster_ordering malloc error");
		exit(0);
	}
}



//read original data from @filename to array @dataBase
/*  
void readDataFromFile()
{
	FILE* fread;
	if( NULL == (fread = fopen(filename, "r")) )
	{
		printf("open file(%s) error\n", filename);
		exit(0);
	}
	int i, j;
	for( i = 1; i <= nspots; i++ )
	{
		for( j = 1; j <= ndim; j++ )
		{
			if( 1 != fscanf(fread, "%lf ", &dataBase[i][j]))
			{
				printf("dataBase fscanf error: (%d, %d)", i, j);
				exit(0);
			}
		}
	}
}
*/ 



float calculateDistance_betweenTwoObject(int ndim, int nspots, float* locs, int Idx_A, int Idx_B)
{
	// calculate distance between two localizations
    float dist = 0.0f;
	int ax = 0;
	for(ax = 0, dist = 0.0f; ax < ndim; ax++)
        dist += (locs[ax * nspots + Idx_A] - locs[ax * nspots + Idx_B]) * (locs[ax * nspots + Idx_A] - locs[ax * nspots + Idx_B]);
	return sqrtf(dist);
}
 


void calculateDistanceMatrix(int ndim, int nspots, float epsilon, float* locs, float* distanceMatrix, int* core_object_size)
{
	// calculate the pairwise-distances
    int i = 0, j = 0;
    float dist = 0.0f;

    // distance matrix
    memset(distanceMatrix, 0, (nspots + 1) * (nspots + 1) * sizeof(float));
    for(i = 1; i < nspots + 1; i++)
		for(j = i + 1; j < nspots + 1; j++)
		{
			dist = calculateDistance_betweenTwoObject(ndim, nspots, locs, i - 1, j - 1);
            distanceMatrix[i * (nspots + 1) + j] = dist;
			distanceMatrix[j * (nspots + 1) + i] = dist;
			if( dist <= epsilon )
			{
				distanceMatrix[i * (nspots + 1)] += 1;
				distanceMatrix[j] += 1;
			}
		}
	
	
    //statistic the number of core object
	for(i = 1; i <= nspots; i++)
		if( distanceMatrix[i * (nspots + 1)] >= MinPts - 1 )
			*core_object_size += 1;
}
 


void createCoreOjectSet()
{
	core_object_set = (Core_Object*)malloc((core_object_size + 1) * sizeof(Core_Object));
	if( !core_object_set )
	{
		printf("core_object_set malloc error!");
		exit(0);
	}
	
    int i = 0, j = 0;
	for(i = 1; i < core_object_size + 1; i++)
	{
		core_object_set[i].directlyDensityReachable_ID = (int*)malloc((INITIAL_SIZE + 1) * sizeof(int));
		if( !core_object_set[i].directlyDensityReachable_ID )
		{
			printf("core_object_set's directlyDensityReachable malloc error\n");
			exit(0);
		}
		core_object_set[i].capacity = INITIAL_SIZE;
		core_object_set[i].size_DDR = 0;
	}
 
	int counter_CO = 1;		//counter_CoreObject
	for(i = 1; i < nspots + 1; i++)
	{
		if(distanceMatrix[i * (nspots + 1)] >= MinPts - 1)
		{
			core_object_set[counter_CO].objectID = i;
			for(j = 1; j <= nspots; j++)
			{
				if( distanceMatrix[i * (nspots + 1) + j] <= epsilon && i != j )
				{
					core_object_set[counter_CO].size_DDR += 1;
					if( core_object_set[counter_CO].size_DDR > core_object_set[counter_CO].capacity )
					{
						core_object_set[counter_CO].directlyDensityReachable_ID = (int*)realloc(core_object_set[counter_CO].directlyDensityReachable_ID, sizeof(int) * (core_object_set[counter_CO].capacity + INCREASEMENT + 1));
						if( !core_object_set[counter_CO].directlyDensityReachable_ID )
						{
							printf("realloc is error: %d ", counter_CO);
							exit(0);
						}
						core_object_set[counter_CO].capacity += INCREASEMENT;
					}
					core_object_set[counter_CO].directlyDensityReachable_ID[core_object_set[counter_CO].size_DDR] = j;
				}
			}
			counter_CO++;
		}
	}
}
 
/*
 * 
 * */
void createObjectNeighborSet()
{
	object_neighbor = (Core_Object*)malloc((nspots + 1) * sizeof(Core_Object));
	if( !object_neighbor )
	{
		printf("object_neighbor malloc error!\n");
		exit(0);
	}
	
    int i = 0, j = 0;
	for( i = 1; i <= nspots; i++ )
	{
		object_neighbor[i].objectID = i;
		object_neighbor[i].size_DDR = 0;
		object_neighbor[i].capacity = INITIAL_SIZE;
		object_neighbor[i].directlyDensityReachable_ID = (int*)malloc(sizeof(int) * (INITIAL_SIZE + 1));
		object_neighbor[i].processed = 0;
		object_neighbor[i].coreDistance = UNDEFINE;
		object_neighbor[i].reachabilityDistance = UNDEFINE;
	}
 
	for( i = 1; i <= nspots; i++ )
	{
		for( j = 1; j <= nspots; j++ )
		{
			if( distanceMatrix[i * (nspots + 1) + j] <= epsilon && i != j )
			{
				object_neighbor[i].size_DDR += 1;
				if( object_neighbor[i].size_DDR > object_neighbor[i].capacity )
				{
					object_neighbor[i].directlyDensityReachable_ID = (int*)realloc(object_neighbor[i].directlyDensityReachable_ID, sizeof(int) * (object_neighbor[i].capacity + INCREASEMENT + 1));
						if( !object_neighbor[i].directlyDensityReachable_ID )
						{
							printf("object_neighbor realloc error: %d\n", i);
							exit(0);
						}
					object_neighbor[i].capacity += INCREASEMENT;
				}
				object_neighbor[i].directlyDensityReachable_ID[object_neighbor[i].size_DDR] = j;
			}
		}
	}
}
 
/****************************************************************************************************
 ****************************************************************************************************
 *                                         OPTICS algorithm
 ****************************************************************************************************
 ****************************************************************************************************/
void OPTICS()
{
	printf("\n\n");
	initialQueue(&orderSeeds);
	int i;
	for( i = 1; i <= nspots; i++ )
	{
		if( isProcessed(i) != 1 )
		{
			printf("\n\n--------------expandClusterOrder: %d------------------\n\n", i);
			ExpandClusterOrder(i);
		}
	}
}




int isProcessed(int objectID)
{
	return object_neighbor[objectID].processed;
}




void ExpandClusterOrder(int objectID)
{
	int currentObjectID;
	object_neighbor[objectID].processed = 1;
	object_neighbor[objectID].reachabilityDistance = UNDEFINE;
	setCoreDistance(objectID);
	saveToClusterOrdering(objectID, cluster_order_counter++);
	if( object_neighbor[objectID].coreDistance != UNDEFINE )
	{
		update(objectID);
		while( !isQueueEmpty(orderSeeds) )
		{
			deleteQueue(&orderSeeds, ¤tObjectID);
			object_neighbor[currentObjectID].processed = 1;
			setCoreDistance(currentObjectID);
			saveToClusterOrdering(currentObjectID, cluster_order_counter++);
			if( object_neighbor[currentObjectID].coreDistance != UNDEFINE )
			{
				update(currentObjectID);
			}
		}
	}
}



void setCoreDistance(int objectID)
{
	float* increase;
	increase = (float*)malloc((object_neighbor[objectID].size_DDR + 1) * sizeof(float));
	if( !increase )
	{
		printf("in function setCoreDistance malloc error\n");
		exit(0);
	}
	
    int i = 0, j = 0;
	for(i = 1; i <= object_neighbor[objectID].size_DDR; i++ )
		increase[i] = distanceMatrix[objectID * (nspots + 1) + object_neighbor[objectID].directlyDensityReachable_ID[i]];
	
	
	int min_index;
	float min_value;
	for(i = 1; i <= object_neighbor[objectID].size_DDR; i++ )
	{
		min_index = i;
		min_value = increase[i];
		for(j = i + 1; j <= object_neighbor[objectID].size_DDR; j++ )
		{
			if( increase[j] < min_value )
			{
				min_index = j;
				min_value = increase[j];
			}
		}
		if( min_index != i )
		{
			increase[0] = increase[min_index];
			increase[min_index] = increase[i];
			increase[i] = increase[0];
		}
	}
 
	if( object_neighbor[objectID].size_DDR >= MinPts - 1 )
		object_neighbor[objectID].coreDistance = increase[MinPts - 1];
    else
		object_neighbor[objectID].coreDistance = UNDEFINE;
	
}



void saveToClusterOrdering(int objectID, int counter)
{
	cluster_ordering[counter].objectID = objectID;
	cluster_ordering[counter].core_distance = object_neighbor[objectID].coreDistance;
	cluster_ordering[counter].reachability_distance = object_neighbor[objectID].reachabilityDistance;
}
 



void update(int centerObjectID)
{
	float new_reachabilityDistance = 0.0;
	int i;
	int objectID;
	for( i = 1; i <= object_neighbor[centerObjectID].size_DDR; i++ )
	{
		objectID = object_neighbor[centerObjectID].directlyDensityReachable_ID[i];
		if( object_neighbor[objectID].processed != 1 )
		{
			new_reachabilityDistance = object_neighbor[centerObjectID].coreDistance > distanceMatrix[centerObjectID * (nspots + 1) + objectID] ? object_neighbor[centerObjectID].coreDistance : distanceMatrix[centerObjectID * (nspots + 1) + objectID];
			if( object_neighbor[objectID].reachabilityDistance == UNDEFINE )
			{
				object_neighbor[objectID].reachabilityDistance = new_reachabilityDistance;
				insertQueue(&orderSeeds, object_neighbor[objectID].objectID, new_reachabilityDistance);
			}
			else
			{
				if( new_reachabilityDistance < object_neighbor[objectID].reachabilityDistance )
				{
					object_neighbor[objectID].reachabilityDistance = new_reachabilityDistance;
					decrease(&orderSeeds, objectID ,new_reachabilityDistance);
				}
			}
		}
	}
}
 
/*
 * write the @cluster_ordering
 * */
void writeClusterOrdering2File()
{
	int i;
	FILE* fwrite;
	if( NULL == (fwrite = fopen(".//OPTICS_ORDERING//cluster_ordering.data", "w")))
	{
		printf("open file(cluster_ordering.data) fail\n");
		exit(0);
	}
	for( i = 1; i <= nspots; i++ )
	{
		fprintf(fwrite, "%d\t%f\t%f\n", cluster_ordering[i].objectID, cluster_ordering[i].core_distance, cluster_ordering[i].reachability_distance);
	}
	fclose(fwrite);
}
 
/*
 * some operation about queue
 * */
void initialQueue(LinkQueue* L)
{
	L->front = (QueueNode*)malloc(sizeof(QueueNode));
	if( !L->front )
	{
		printf("queue optics_init malloc error\n");
		exit(0);
	}
	L->rear = L->front;
	L->rear->next = NULL;
}



void insertQueue(LinkQueue* L, int objectID, float distance)
{
	QueueNode* new = (QueueNode*)malloc(sizeof(QueueNode));
	if( !new )
	{
		printf("insert Queue malloc error");
		exit(0);
	}
	new->data = objectID;
	new->reachabilityDistance = distance;
 
	if( isQueueEmpty(*L) )
	{
		new->next = NULL;
		L->rear->next = new;
		L->rear = new;
	}else	//non-empty
	{
		QueueNode* seek = L->front->next;
		QueueNode* later = L->front;
		while( seek != NULL && seek->reachabilityDistance < distance )
		{
			later = seek;
			seek = seek->next;
		}
		if( seek == NULL )
		{
			new->next = L->rear->next;
			L->rear->next = new;
			L->rear = new;
		}else
		{
			new->next = seek;
			later->next = new;
		}
	}
}



void decrease(LinkQueue* L, int objectID, float distance)
{
	QueueNode* seek = L->front->next;
	QueueNode* follower = L->front;
	while( seek->data != objectID )
	{
		follower = seek;
		seek = seek->next;
	}
	if( seek == L->rear )
	{
		follower->next = NULL;
		L->rear = follower;
	}else
	{
		follower->next = seek->next;
	}
	QueueNode* new = (QueueNode*)malloc(sizeof(QueueNode));
	new->data = objectID;
	new->reachabilityDistance = distance;
	seek = L->front->next;
	follower = L->front;
	//printf("c");
	if( isQueueEmpty(*L) == 1 )
	{
		L->front->next = new;
		L->rear = new;
		L->rear->next = NULL;
		//printf("d");
	}else
	{
		//printf("e");
		while( seek != NULL && seek->reachabilityDistance < distance )
		{
			follower = seek;
			seek = seek->next;
		}
		if( seek == NULL )
		{
			//printf("f");
			follower->next = new;
			new->next = NULL;
			L->rear = new;
		}else
		{
			//printf("g");
			new->next = seek;
			follower->next = new;
		}
	}
}



void deleteQueue(LinkQueue* L, int* objectID)
{
	if( isQueueEmpty(*L) )
	{
		printf("\tQueue is empty\n");
		exit(0);
	}
	QueueNode* f = L->front->next;
	*objectID = f->data;
	L->front->next = f->next;
	if( f == L->rear )
	{
		L->rear = L->front;
	}
	free(f);
}



int isQueueEmpty(LinkQueue L)
{
	return L.rear == L.front ? 1 : 0;
}



void printQueue(LinkQueue L)
{
	printf("\nqueue operation : print\n");
	if( isQueueEmpty(L) )
	{
		printf("\tQueue is empty\n");
	}
	else
	{
		L.front = L.front->next;
		while( L.front != L.rear )
		{
			printf("%d: %f\n", L.front->data, L.front->reachabilityDistance);
			L.front = L.front->next;
		}
		printf("%d: %f \n\n", L.front->data, L.front->reachabilityDistance);
	}
}



void testQueue()
{
	LinkQueue L;
	int x;
	initialQueue(&L);
	insertQueue(&L, 4, 1.254);
	insertQueue(&L, 2, 2.2564);
	insertQueue(&L, 6, 1.0254);
	insertQueue(&L, 1, 0.56);
	insertQueue(&L, 5, 4.15);
	insertQueue(&L, 3, 2.342);
	insertQueue(&L, 7, 2.0214);
	insertQueue(&L, 8, 0.254);
	printQueue(L);
 
	decrease(&L, 3, 0.50);
	printQueue(L);
	
	decrease(&L, 5, 4.00);
	printQueue(L);
}