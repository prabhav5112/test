MatrixMatrixMultiply (int n, double *a, double *b, double *c, MPI_Comm comm)
{
  int i;
  int nlocal;
  int npes, dims[2], periods[2];
  int myrank, my2drank, mycoords[2];
  int uprank, downrank, leftrank, rightrank, coords[2];
  int shiftsource, shiftdest;
  MPI_Status status;
  MPI_Comm comm_2d;		/* Get the communicator-related information */
  MPI_Comm_size (comm, &npes);
  MPI_Comm_rank (comm, &myrank);
/* Set up the Cartesian topology */
  dims[0] = dims[1] = sqrt (npes);
/* Set the periods for wraparound connections */
  periods[0] = periods[1] = 1;
/* Create the Cartesian topology, with rank reordering */
  MPI_Cart_create (comm, 2, dims, periods, 1, &comm_2d);
/* Get the rank and coordinates with respect to the new topology */
  MPI_Comm_rank (comm_2d, &my2drank);
  MPI_Cart_coords (comm_2d, my2drank, 2, mycoords);
/* Compute ranks of the up and left shifts */
  MPI_Cart_shift (comm_2d, 0, -1, &rightrank, &leftrank);
  MPI_Cart_shift (comm_2d, 1, -1, &downrank, &uprank);
/* Determine the dimension of the local matrix block */
  nlocal = n / dims[0];
/* Perform the initial matrix alignment. First for A and then for B */
  MPI_Cart_shift (comm_2d, 0, -mycoords[0], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (a, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
			shiftsource, 1, comm_2d, &status);
  MPI_Cart_shift (comm_2d, 1, -mycoords[1], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (b, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
			shiftsource, 1, comm_2d, &status);
/* Get into the main computation loop */
  for (i = 0; i < dims[0]; i++)
    {
      MatrixMultiply (nlocal, a, b, c);	/* c = c + a * b *//* Shift matrix A left by one */
      MPI_Sendrecv_replace (a, nlocal * nlocal, MPI_DOUBLE, leftrank, 1,
			    rightrank, 1, comm_2d, &status);
/* Shift matrix B up by one */
      MPI_Sendrecv_replace (b, nlocal * nlocal, MPI_DOUBLE, uprank, 1,
			    downrank, 1, comm_2d, &status);
    }
/* Restore the original distribution of A and B */
  MPI_Cart_shift (comm_2d, 0, +mycoords[0], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (a, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
			shiftsource, 1, comm_2d, &status);
  MPI_Cart_shift (comm_2d, 1, +mycoords[1], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (b, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
			shiftsource, 1, comm_2d, &status);
  MPI_Comm_free (&comm_2d);	/* Free up communicator */
}

/* This function performs a serial matrix-matrix multiplication c = a * b */
MatrixMultiply (int n, double *a, double *b, double *c)
{
  int i, j, k;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
	c[i * n + j] += a[i * n + k] * b[k * n + j];
}


void
MatrixMatrixMultiply_NonBlocking (int n, double *a, double *b, double *c,
				  MPI_Comm comm)
{
  int i, j, nlocal;
  double *a_buffers[2], *b_buffers[2];
  int npes, dims[2], periods[2];
  int myrank, my2drank, mycoords[2];
  int uprank, downrank, leftrank, rightrank, coords[2];
  int shiftsource, shiftdest;
  MPI_Status status;
  MPI_Comm comm_2d;
  MPI_Request reqs[4];
/* Get the communicator related information */
  MPI_Comm_size (comm, &npes);
  MPI_Comm_rank (comm, &myrank);
/* Set up the Cartesian topology */ dims[0] = dims[1] = sqrt (npes);
/* Set the periods for wraparound connections */
  periods[0] = periods[1] = 1;
/* Create the Cartesian topology, with rank reordering */
  MPI_Cart_create (comm, 2, dims, periods, 1, &comm_2d);
/* Get the rank and coordinates with respect to the new topology */
  MPI_Comm_rank (comm_2d, &my2drank);
  MPI_Cart_coords (comm_2d, my2drank, 2, mycoords);
/* Compute ranks of the up and left shifts */
  MPI_Cart_shift (comm_2d, 0, -1, &rightrank, &leftrank);
  MPI_Cart_shift (comm_2d, 1, -1, &downrank, &uprank);
/* Determine the dimension of the local matrix block */
  nlocal = n / dims[0];
/* Setup the a_buffers and b_buffers arrays */
  a_buffers[0] = a;
  a_buffers[1] = (double *) malloc (nlocal * nlocal * sizeof (double));
  b_buffers[0] = b;
  b_buffers[1] = (double *) malloc (nlocal * nlocal * sizeof (double));
/* Perform the initial matrix alignment. First for A and then for B */
  MPI_Cart_shift (comm_2d, 0, -mycoords[0], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (a_buffers[0], nlocal * nlocal, MPI_DOUBLE, shiftdest,
			1, shiftsource, 1, comm_2d, &status);
  MPI_Cart_shift (comm_2d, 1, -mycoords[1], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (b_buffers[0], nlocal * nlocal, MPI_DOUBLE, shiftdest,
			1, shiftsource, 1, comm_2d, &status);
/* Get into the main computation loop */
  for (i = 0; i < dims[0]; i++)
    {
      MPI_Isend (a_buffers[i % 2], nlocal * nlocal, MPI_DOUBLE, leftrank, 1,
		 comm_2d, &reqs[0]);
      MPI_Isend (b_buffers[i % 2], nlocal * nlocal, MPI_DOUBLE, uprank, 1,
		 comm_2d, &reqs[1]);
      MPI_Irecv (a_buffers[(i + 1) % 2], nlocal * nlocal, MPI_DOUBLE,
		 rightrank, 1, comm_2d, &reqs[2]);
      MPI_Irecv (b_buffers[(i + 1) % 2], nlocal * nlocal, MPI_DOUBLE,
		 downrank, 1, comm_2d, &reqs[3]);
/* c = c + a*b */
      MatrixMultiply (nlocal, a_buffers[i % 2], b_buffers[i % 2], c);
      for (j = 0; j < 4; j++)
	MPI_Wait (&reqs[j], &status);
    }
/* Restore the original distribution of a and b */
  MPI_Cart_shift (comm_2d, 0, +mycoords[0], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (a_buffers[i % 2], nlocal * nlocal, MPI_DOUBLE,
			shiftdest, 1, shiftsource, 1, comm_2d, &status);
  MPI_Cart_shift (comm_2d, 1, +mycoords[1], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace (b_buffers[i % 2], nlocal * nlocal, MPI_DOUBLE,
			shiftdest, 1, shiftsource, 1, comm_2d, &status);
/* Free up communicator and allocated memory */
  MPI_Comm_free (&comm_2d);
  free (a_buffers[1]);
  free (b_buffers[1]);
}

int
MPI_Barrier (MPI_Comm comm)
     int MPI_Bcast (void *buf, int count, MPI_Datatype datatype,
		    int source, MPI_Comm comm)
     int MPI_Reduce (void *sendbuf, void *recvbuf, int count,
		     MPI_Datatype datatype, MPI_Op op, int target,
		     MPI_Comm comm)
/* MPI_Reduce combines elements from sendbuf across all processes using the
operation specified in op.
The result is stored in the recvbuf of the process with rank target. */
     int MPI_Allreduce (void *sendbuf, void *recvbuf, int count,
			MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
/* All processes receive the result */
     int MPI_Scan (void *sendbuf, void *recvbuf, int count,
		   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
/* ●​ MPI_Scan performs a prefix reduction on sendbuf and stores the result in recvbuf.
●​ Process i receives the reduction result of all processes from rank 0 to i. */
     int MPI_Gather (void *sendbuf, int sendcount,
		     MPI_Datatype senddatatype, void *recvbuf, int recvcount,
		     MPI_Datatype recvdatatype, int target, MPI_Comm comm)
     int MPI_Allgather (void *sendbuf, int sendcount,
			MPI_Datatype senddatatype, void *recvbuf,
			int recvcount, MPI_Datatype recvdatatype,
			MPI_Comm comm)
     int MPI_Gatherv (void *sendbuf, int sendcount, MPI_Datatype senddatatype,
		      void *recvbuf, int *recvcounts, int *displs,
		      MPI_Datatype recvdatatype, int target, MPI_Comm comm)
     int MPI_Allgatherv (void *sendbuf, int sendcount,
			 MPI_Datatype senddatatype, void *recvbuf,
			 int *recvcounts, int *displs,
			 MPI_Datatype recvdatatype, MPI_Comm comm)
     int MPI_Scatterv (void *sendbuf, int *sendcounts, int *displs,
		       MPI_Datatype senddatatype, void *recvbuf,
		       int recvcount, MPI_Datatype recvdatatype, int source,
		       MPI_Comm comm)
/* sendcounts[i] specifies how many elements process i receives.
displs[i] specifies the starting position of each block. */
     int MPI_Alltoall (void *sendbuf, int sendcount,
		       MPI_Datatype senddatatype, void *recvbuf,
		       int recvcount, MPI_Datatype recvdatatype,
		       MPI_Comm comm)
     int MPI_Alltoallv (void *sendbuf, int *sendcounts, int *sdispls,
			MPI_Datatype senddatatype, void *recvbuf,
			int *recvcounts, int *rdispls,
			MPI_Datatype recvdatatype, MPI_Comm comm)
/* sendcounts[i] specifies the number of elements sent to process i.
sdispls[i] specifies the location of the elements in sendbuf.
recvcounts[i] specifies the number of elements received from process i.
rdispls[i] specifies where the received elements are stored. */
RowMatrixVectorMultiply (int n, double *a, double *b, double *x,
			 MPI_Comm comm)
{
  int i, j;
  int nlocal;
/* Number of locally stored rows of A */
  double *fb;
/* Will point to a buffer that stores the entire vector b */
  int npes, myrank;
  MPI_Status status;
/* Get information about the communicator */
  MPI_Comm_size (comm, &npes);
  MPI_Comm_rank (comm, &myrank);
/* Allocate the memory that will store the entire vector b */
  fb = (double *) malloc (n * sizeof (double));
  nlocal = n / npes;
/* Gather the entire vector b on each processor using MPI's ALLGATHER operation */
  MPI_Allgather (b, nlocal, MPI_DOUBLE, fb, nlocal, MPI_DOUBLE, comm);
/* Perform the matrix-vector multiplication involving the locally stored submatrix */
  for (i = 0; i < nlocal; i++)
    {
      x[i] = 0.0;
      for (j = 0; j < n; j++)
	x[i] += a[i * n + j] * fb[j];
    }
  free (fb);
}

ColMatrixVectorMultiply (int n, double *a, double *b, double *x,
			 MPI_Comm comm)
{
  int i, j;
  int nlocal;
  double *px;
  double *fx;
  int npes, myrank;
  MPI_Status status;
/* Get identity and size information from the communicator */
  MPI_Comm_size (comm, &npes);
  MPI_Comm_rank (comm, &myrank);
  nlocal = n / npes;
/* Allocate memory for arrays storing intermediate results. */
  px = (double *) malloc (n * sizeof (double));
  fx = (double *) malloc (n * sizeof (double));
/* Compute the partial-dot products that correspond to the local columns of A. */
  for (i = 0; i < n; i++)
    {
      px[i] = 0.0;
      for (j = 0; j < nlocal; j++)
	px[i] += a[i * nlocal + j] * b[j];
    }
/* Sum-up the results by performing an element-wise reduction operation */
  MPI_Reduce (px, fx, n, MPI_DOUBLE, MPI_SUM, 0, comm);
/* Redistribute fx in a fashion similar to that of vector b */
  MPI_Scatter (fx, nlocal, MPI_DOUBLE, x, nlocal, MPI_DOUBLE, 0, comm);
  free (px);
  free (fx);
}

int *
SampleSort (int n, int *elmnts, int *nsorted, MPI_Comm comm)
{
  int i, j, nlocal, npes, myrank;
  int *sorted_elmnts, *splitters, *allpicks;
  int *scounts, *sdispls, *rcounts, *rdispls;
/* Get communicator-related information */
  MPI_Comm_size (comm, &npes);
  MPI_Comm_rank (comm, &myrank);
  nlocal = n / npes;
/* Allocate memory for the arrays that will store the splitters */
  splitters = (int *) malloc (npes * sizeof (int));
  allpicks = (int *) malloc (npes * (npes - 1) * sizeof (int));
/* Sort local array */
  qsort (elmnts, nlocal, sizeof (int), IncOrder);
/* Select local npes-1 equally spaced elements */
  for (i = 1; i < npes; i++)
    splitters[i - 1] = elmnts[i * nlocal / npes];
/* Gather the samples in the processors */
  MPI_Allgather (splitters, npes - 1, MPI_INT, allpicks, npes - 1, MPI_INT,
		 comm);
/* Sort these samples */
  qsort (allpicks, npes * (npes - 1), sizeof (int), IncOrder);
/* Select splitters */
  for (i = 1; i < npes; i++)
    splitters[i - 1] = allpicks[i * npes];
  splitters[npes - 1] = MAXINT;
/* Compute the number of elements that belong to each bucket */
  scounts = (int *) malloc (npes * sizeof (int));
  for (i = 0; i < npes; i++)
    scounts[i] = 0;
  for (j = i = 0; i < nlocal; i++)
    {
      if (elmnts[i] < splitters[j])
	scounts[j]++;
      else
	scounts[++j]++;
    }
/* Determine the starting location of each bucket's elements in the elmnts array */
  sdispls = (int *) malloc (npes * sizeof (int));
  sdispls[0] = 0;
  for (i = 1; i < npes; i++)
    sdispls[i] = sdispls[i - 1] + scounts[i - 1];
/* Perform an all-to-all to inform the corresponding processes of the number of elements */
/* they are going to receive. This information is stored in rcounts array */
  rcounts = (int *) malloc (npes * sizeof (int));
  MPI_Alltoall (scounts, 1, MPI_INT, rcounts, 1, MPI_INT, comm);
/* Based on rcounts determine where in the local array the data from each processor */
/* will be stored. This array will store the received elements as well as the final sorted sequence */
  rdispls = (int *) malloc (npes * sizeof (int));
  rdispls[0] = 0;
  for (i = 1; i < npes; i++)
    rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
  *nsorted = rdispls[npes - 1] + rcounts[i - 1];
  sorted_elmnts = (int *) malloc ((*nsorted) * sizeof (int));
/* Each process sends and receives the corresponding elements, using the MPI_Alltoallv */
/* operation. The arrays scounts and sdispls are used to specify the number of elements */
/* to be sent and where these elements are stored, respectively. The arrays rcounts */
/* and rdispls are used to specify the number of elements to be received, and where these */
/* elements will be stored, respectively. */
  MPI_Alltoallv (elmnts, scounts, sdispls, MPI_INT, sorted_elmnts, rcounts,
		 rdispls, MPI_INT, comm);
/* Perform the final local sort */
  qsort (sorted_elmnts, *nsorted, sizeof (int), IncOrder);
  free (splitters);
  free (allpicks);
  free (scounts);
  free (sdispls);
  free (rcounts);
  free (rdispls);
  return sorted_elmnts;
}

/* This function is collective, meaning all processes in comm must call it.
The function partitions comm into disjoint subgroups based on the color parameter.
Processes within the same subgroup are ranked according to the key parameter.
The new communicator for each subgroup is stored in newcomm. */
int
MPI_Comm_split (MPI_Comm comm, int color, int key, MPI_Comm * newcomm)
/*The keep_dims array determines which dimensions are retained in the new
sub-topology.
If keep_dims[i] is true, the i-th dimension is retained; otherwise, it is not.
The number of sub-topologies created is equal to the product of the number of
processes along the discarded dimensions.*/
     int MPI_Cart_sub (MPI_Comm comm_cart, int *keep_dims,
		       MPI_Comm *
		       comm_subcart) MatrixVectorMultiply_2D (int n,
							      double *a,
							      double *b,
							      double *x,
							      MPI_Comm comm)
{
  int ROW = 0, COL = 1;		/* Improve readability */
  int i, j, nlocal;
  double *px;			/* Will store partial dot products */
  int npes, dims[2], periods[2], keep_dims[2];
  int myrank, my2drank, mycoords[2];
  int other_rank, coords[2];
  MPI_Status status;
  MPI_Comm comm_2d, comm_row, comm_col;
/* Get information about the communicator */
  MPI_Comm_size (comm, &npes);
  MPI_Comm_rank (comm, &myrank);
/* Compute the size of the square grid */
  dims[ROW] = dims[COL] = sqrt (npes);
  nlocal = n / dims[ROW];
/* Allocate memory for the array that will hold the partial dot-products */
  px = malloc (nlocal * sizeof (double));	/* Set up the Cartesian topology and get the rank & coordinates of the process in this
						   topology */
  periods[ROW] = periods[COL] = 1;	/* Set the periods for wrap-around connections */
  MPI_Cart_create (MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);
  MPI_Comm_rank (comm_2d, &my2drank);	/* Get my rank in the new topology */
  MPI_Cart_coords (comm_2d, my2drank, 2, mycoords);	/* Get my coordinates */
/* Create the row-based sub-topology */
  keep_dims[ROW] = 0;
  keep_dims[COL] = 1;
  MPI_Cart_sub (comm_2d, keep_dims, &comm_row);
/* Create the column-based sub-topology */
  keep_dims[ROW] = 1;
  keep_dims[COL] = 0;
  MPI_Cart_sub (comm_2d, keep_dims, &comm_col);
/* Redistribute the b vector. */
/* Step 1. The processors along the 0th column send their data to the diagonal processors */
  if (mycoords[COL] == 0 && mycoords[ROW] != 0)
    {				/* I'm in the first column */
      coords[ROW] = mycoords[ROW];
      coords[COL] = mycoords[ROW];
      MPI_Cart_rank (comm_2d, coords, &other_rank);
      MPI_Send (b, nlocal, MPI_DOUBLE, other_rank, 1, comm_2d);
    }
  if (mycoords[ROW] == mycoords[COL] && mycoords[ROW] != 0)
    {
      coords[ROW] = mycoords[ROW];
      coords[COL] = 0;
      MPI_Cart_rank (comm_2d, coords, &other_rank);
      MPI_Recv (b, nlocal, MPI_DOUBLE, other_rank, 1, comm_2d, &status);
    }
/* Step 2. The diagonal processors perform a column-wise broadcast */
  coords[0] = mycoords[COL];
  MPI_Cart_rank (comm_col, coords, &other_rank);
  MPI_Bcast (b, nlocal, MPI_DOUBLE, other_rank, comm_col);
/* Get into the main computational loop */
  for (i = 0; i < nlocal; i++)
    {
      px[i] = 0.0;
      for (j = 0; j < nlocal; j++)
	px[i] += a[i * nlocal + j] * b[j];
    }
/* Perform the sum-reduction along the rows to add up the partial dot-products */
  coords[0] = 0;
  MPI_Cart_rank (comm_row, coords, &other_rank);
  MPI_Reduce (px, x, nlocal, MPI_DOUBLE, MPI_SUM, other_rank, comm_row);
  MPI_Comm_free (&comm_2d);	/* Free up communicator */
  MPI_Comm_free (&comm_row);	/* Free up communicator */
  MPI_Comm_free (&comm_col);	/* Free up communicator */
  free (px);
}
