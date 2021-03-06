??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
cnn__gender_3/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name cnn__gender_3/conv2d_15/kernel
?
2cnn__gender_3/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpcnn__gender_3/conv2d_15/kernel*&
_output_shapes
: *
dtype0
?
cnn__gender_3/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namecnn__gender_3/conv2d_15/bias
?
0cnn__gender_3/conv2d_15/bias/Read/ReadVariableOpReadVariableOpcnn__gender_3/conv2d_15/bias*
_output_shapes
: *
dtype0
?
*cnn__gender_3/batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*cnn__gender_3/batch_normalization_15/gamma
?
>cnn__gender_3/batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOp*cnn__gender_3/batch_normalization_15/gamma*
_output_shapes
: *
dtype0
?
)cnn__gender_3/batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)cnn__gender_3/batch_normalization_15/beta
?
=cnn__gender_3/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOp)cnn__gender_3/batch_normalization_15/beta*
_output_shapes
: *
dtype0
?
0cnn__gender_3/batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20cnn__gender_3/batch_normalization_15/moving_mean
?
Dcnn__gender_3/batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp0cnn__gender_3/batch_normalization_15/moving_mean*
_output_shapes
: *
dtype0
?
4cnn__gender_3/batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64cnn__gender_3/batch_normalization_15/moving_variance
?
Hcnn__gender_3/batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp4cnn__gender_3/batch_normalization_15/moving_variance*
_output_shapes
: *
dtype0
?
cnn__gender_3/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name cnn__gender_3/conv2d_16/kernel
?
2cnn__gender_3/conv2d_16/kernel/Read/ReadVariableOpReadVariableOpcnn__gender_3/conv2d_16/kernel*&
_output_shapes
: @*
dtype0
?
cnn__gender_3/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecnn__gender_3/conv2d_16/bias
?
0cnn__gender_3/conv2d_16/bias/Read/ReadVariableOpReadVariableOpcnn__gender_3/conv2d_16/bias*
_output_shapes
:@*
dtype0
?
*cnn__gender_3/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*cnn__gender_3/batch_normalization_16/gamma
?
>cnn__gender_3/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOp*cnn__gender_3/batch_normalization_16/gamma*
_output_shapes
:@*
dtype0
?
)cnn__gender_3/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)cnn__gender_3/batch_normalization_16/beta
?
=cnn__gender_3/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOp)cnn__gender_3/batch_normalization_16/beta*
_output_shapes
:@*
dtype0
?
0cnn__gender_3/batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20cnn__gender_3/batch_normalization_16/moving_mean
?
Dcnn__gender_3/batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp0cnn__gender_3/batch_normalization_16/moving_mean*
_output_shapes
:@*
dtype0
?
4cnn__gender_3/batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64cnn__gender_3/batch_normalization_16/moving_variance
?
Hcnn__gender_3/batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp4cnn__gender_3/batch_normalization_16/moving_variance*
_output_shapes
:@*
dtype0
?
cnn__gender_3/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*/
shared_name cnn__gender_3/conv2d_17/kernel
?
2cnn__gender_3/conv2d_17/kernel/Read/ReadVariableOpReadVariableOpcnn__gender_3/conv2d_17/kernel*'
_output_shapes
:@?*
dtype0
?
cnn__gender_3/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namecnn__gender_3/conv2d_17/bias
?
0cnn__gender_3/conv2d_17/bias/Read/ReadVariableOpReadVariableOpcnn__gender_3/conv2d_17/bias*
_output_shapes	
:?*
dtype0
?
*cnn__gender_3/batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*cnn__gender_3/batch_normalization_17/gamma
?
>cnn__gender_3/batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOp*cnn__gender_3/batch_normalization_17/gamma*
_output_shapes	
:?*
dtype0
?
)cnn__gender_3/batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)cnn__gender_3/batch_normalization_17/beta
?
=cnn__gender_3/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOp)cnn__gender_3/batch_normalization_17/beta*
_output_shapes	
:?*
dtype0
?
0cnn__gender_3/batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20cnn__gender_3/batch_normalization_17/moving_mean
?
Dcnn__gender_3/batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp0cnn__gender_3/batch_normalization_17/moving_mean*
_output_shapes	
:?*
dtype0
?
4cnn__gender_3/batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*E
shared_name64cnn__gender_3/batch_normalization_17/moving_variance
?
Hcnn__gender_3/batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp4cnn__gender_3/batch_normalization_17/moving_variance*
_output_shapes	
:?*
dtype0
?
cnn__gender_3/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *.
shared_namecnn__gender_3/dense_15/kernel
?
1cnn__gender_3/dense_15/kernel/Read/ReadVariableOpReadVariableOpcnn__gender_3/dense_15/kernel* 
_output_shapes
:
?? *
dtype0
?
cnn__gender_3/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecnn__gender_3/dense_15/bias
?
/cnn__gender_3/dense_15/bias/Read/ReadVariableOpReadVariableOpcnn__gender_3/dense_15/bias*
_output_shapes
: *
dtype0
?
cnn__gender_3/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_namecnn__gender_3/dense_16/kernel
?
1cnn__gender_3/dense_16/kernel/Read/ReadVariableOpReadVariableOpcnn__gender_3/dense_16/kernel*
_output_shapes

: *
dtype0
?
cnn__gender_3/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecnn__gender_3/dense_16/bias
?
/cnn__gender_3/dense_16/bias/Read/ReadVariableOpReadVariableOpcnn__gender_3/dense_16/bias*
_output_shapes
:*
dtype0
?
cnn__gender_3/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namecnn__gender_3/dense_17/kernel
?
1cnn__gender_3/dense_17/kernel/Read/ReadVariableOpReadVariableOpcnn__gender_3/dense_17/kernel*
_output_shapes

:*
dtype0
?
cnn__gender_3/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecnn__gender_3/dense_17/bias
?
/cnn__gender_3/dense_17/bias/Read/ReadVariableOpReadVariableOpcnn__gender_3/dense_17/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
%Adam/cnn__gender_3/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/cnn__gender_3/conv2d_15/kernel/m
?
9Adam/cnn__gender_3/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/cnn__gender_3/conv2d_15/kernel/m*&
_output_shapes
: *
dtype0
?
#Adam/cnn__gender_3/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/cnn__gender_3/conv2d_15/bias/m
?
7Adam/cnn__gender_3/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOp#Adam/cnn__gender_3/conv2d_15/bias/m*
_output_shapes
: *
dtype0
?
1Adam/cnn__gender_3/batch_normalization_15/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31Adam/cnn__gender_3/batch_normalization_15/gamma/m
?
EAdam/cnn__gender_3/batch_normalization_15/gamma/m/Read/ReadVariableOpReadVariableOp1Adam/cnn__gender_3/batch_normalization_15/gamma/m*
_output_shapes
: *
dtype0
?
0Adam/cnn__gender_3/batch_normalization_15/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/cnn__gender_3/batch_normalization_15/beta/m
?
DAdam/cnn__gender_3/batch_normalization_15/beta/m/Read/ReadVariableOpReadVariableOp0Adam/cnn__gender_3/batch_normalization_15/beta/m*
_output_shapes
: *
dtype0
?
%Adam/cnn__gender_3/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%Adam/cnn__gender_3/conv2d_16/kernel/m
?
9Adam/cnn__gender_3/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/cnn__gender_3/conv2d_16/kernel/m*&
_output_shapes
: @*
dtype0
?
#Adam/cnn__gender_3/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/cnn__gender_3/conv2d_16/bias/m
?
7Adam/cnn__gender_3/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOp#Adam/cnn__gender_3/conv2d_16/bias/m*
_output_shapes
:@*
dtype0
?
1Adam/cnn__gender_3/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31Adam/cnn__gender_3/batch_normalization_16/gamma/m
?
EAdam/cnn__gender_3/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp1Adam/cnn__gender_3/batch_normalization_16/gamma/m*
_output_shapes
:@*
dtype0
?
0Adam/cnn__gender_3/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/cnn__gender_3/batch_normalization_16/beta/m
?
DAdam/cnn__gender_3/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp0Adam/cnn__gender_3/batch_normalization_16/beta/m*
_output_shapes
:@*
dtype0
?
%Adam/cnn__gender_3/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*6
shared_name'%Adam/cnn__gender_3/conv2d_17/kernel/m
?
9Adam/cnn__gender_3/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/cnn__gender_3/conv2d_17/kernel/m*'
_output_shapes
:@?*
dtype0
?
#Adam/cnn__gender_3/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/cnn__gender_3/conv2d_17/bias/m
?
7Adam/cnn__gender_3/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOp#Adam/cnn__gender_3/conv2d_17/bias/m*
_output_shapes	
:?*
dtype0
?
1Adam/cnn__gender_3/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31Adam/cnn__gender_3/batch_normalization_17/gamma/m
?
EAdam/cnn__gender_3/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp1Adam/cnn__gender_3/batch_normalization_17/gamma/m*
_output_shapes	
:?*
dtype0
?
0Adam/cnn__gender_3/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/cnn__gender_3/batch_normalization_17/beta/m
?
DAdam/cnn__gender_3/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp0Adam/cnn__gender_3/batch_normalization_17/beta/m*
_output_shapes	
:?*
dtype0
?
$Adam/cnn__gender_3/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *5
shared_name&$Adam/cnn__gender_3/dense_15/kernel/m
?
8Adam/cnn__gender_3/dense_15/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/cnn__gender_3/dense_15/kernel/m* 
_output_shapes
:
?? *
dtype0
?
"Adam/cnn__gender_3/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/cnn__gender_3/dense_15/bias/m
?
6Adam/cnn__gender_3/dense_15/bias/m/Read/ReadVariableOpReadVariableOp"Adam/cnn__gender_3/dense_15/bias/m*
_output_shapes
: *
dtype0
?
$Adam/cnn__gender_3/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *5
shared_name&$Adam/cnn__gender_3/dense_16/kernel/m
?
8Adam/cnn__gender_3/dense_16/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/cnn__gender_3/dense_16/kernel/m*
_output_shapes

: *
dtype0
?
"Adam/cnn__gender_3/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/cnn__gender_3/dense_16/bias/m
?
6Adam/cnn__gender_3/dense_16/bias/m/Read/ReadVariableOpReadVariableOp"Adam/cnn__gender_3/dense_16/bias/m*
_output_shapes
:*
dtype0
?
$Adam/cnn__gender_3/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/cnn__gender_3/dense_17/kernel/m
?
8Adam/cnn__gender_3/dense_17/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/cnn__gender_3/dense_17/kernel/m*
_output_shapes

:*
dtype0
?
"Adam/cnn__gender_3/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/cnn__gender_3/dense_17/bias/m
?
6Adam/cnn__gender_3/dense_17/bias/m/Read/ReadVariableOpReadVariableOp"Adam/cnn__gender_3/dense_17/bias/m*
_output_shapes
:*
dtype0
?
%Adam/cnn__gender_3/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/cnn__gender_3/conv2d_15/kernel/v
?
9Adam/cnn__gender_3/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/cnn__gender_3/conv2d_15/kernel/v*&
_output_shapes
: *
dtype0
?
#Adam/cnn__gender_3/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/cnn__gender_3/conv2d_15/bias/v
?
7Adam/cnn__gender_3/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOp#Adam/cnn__gender_3/conv2d_15/bias/v*
_output_shapes
: *
dtype0
?
1Adam/cnn__gender_3/batch_normalization_15/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31Adam/cnn__gender_3/batch_normalization_15/gamma/v
?
EAdam/cnn__gender_3/batch_normalization_15/gamma/v/Read/ReadVariableOpReadVariableOp1Adam/cnn__gender_3/batch_normalization_15/gamma/v*
_output_shapes
: *
dtype0
?
0Adam/cnn__gender_3/batch_normalization_15/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/cnn__gender_3/batch_normalization_15/beta/v
?
DAdam/cnn__gender_3/batch_normalization_15/beta/v/Read/ReadVariableOpReadVariableOp0Adam/cnn__gender_3/batch_normalization_15/beta/v*
_output_shapes
: *
dtype0
?
%Adam/cnn__gender_3/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%Adam/cnn__gender_3/conv2d_16/kernel/v
?
9Adam/cnn__gender_3/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/cnn__gender_3/conv2d_16/kernel/v*&
_output_shapes
: @*
dtype0
?
#Adam/cnn__gender_3/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/cnn__gender_3/conv2d_16/bias/v
?
7Adam/cnn__gender_3/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOp#Adam/cnn__gender_3/conv2d_16/bias/v*
_output_shapes
:@*
dtype0
?
1Adam/cnn__gender_3/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31Adam/cnn__gender_3/batch_normalization_16/gamma/v
?
EAdam/cnn__gender_3/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp1Adam/cnn__gender_3/batch_normalization_16/gamma/v*
_output_shapes
:@*
dtype0
?
0Adam/cnn__gender_3/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/cnn__gender_3/batch_normalization_16/beta/v
?
DAdam/cnn__gender_3/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp0Adam/cnn__gender_3/batch_normalization_16/beta/v*
_output_shapes
:@*
dtype0
?
%Adam/cnn__gender_3/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*6
shared_name'%Adam/cnn__gender_3/conv2d_17/kernel/v
?
9Adam/cnn__gender_3/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/cnn__gender_3/conv2d_17/kernel/v*'
_output_shapes
:@?*
dtype0
?
#Adam/cnn__gender_3/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/cnn__gender_3/conv2d_17/bias/v
?
7Adam/cnn__gender_3/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOp#Adam/cnn__gender_3/conv2d_17/bias/v*
_output_shapes	
:?*
dtype0
?
1Adam/cnn__gender_3/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*B
shared_name31Adam/cnn__gender_3/batch_normalization_17/gamma/v
?
EAdam/cnn__gender_3/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp1Adam/cnn__gender_3/batch_normalization_17/gamma/v*
_output_shapes	
:?*
dtype0
?
0Adam/cnn__gender_3/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/cnn__gender_3/batch_normalization_17/beta/v
?
DAdam/cnn__gender_3/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp0Adam/cnn__gender_3/batch_normalization_17/beta/v*
_output_shapes	
:?*
dtype0
?
$Adam/cnn__gender_3/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?? *5
shared_name&$Adam/cnn__gender_3/dense_15/kernel/v
?
8Adam/cnn__gender_3/dense_15/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/cnn__gender_3/dense_15/kernel/v* 
_output_shapes
:
?? *
dtype0
?
"Adam/cnn__gender_3/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/cnn__gender_3/dense_15/bias/v
?
6Adam/cnn__gender_3/dense_15/bias/v/Read/ReadVariableOpReadVariableOp"Adam/cnn__gender_3/dense_15/bias/v*
_output_shapes
: *
dtype0
?
$Adam/cnn__gender_3/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *5
shared_name&$Adam/cnn__gender_3/dense_16/kernel/v
?
8Adam/cnn__gender_3/dense_16/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/cnn__gender_3/dense_16/kernel/v*
_output_shapes

: *
dtype0
?
"Adam/cnn__gender_3/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/cnn__gender_3/dense_16/bias/v
?
6Adam/cnn__gender_3/dense_16/bias/v/Read/ReadVariableOpReadVariableOp"Adam/cnn__gender_3/dense_16/bias/v*
_output_shapes
:*
dtype0
?
$Adam/cnn__gender_3/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/cnn__gender_3/dense_17/kernel/v
?
8Adam/cnn__gender_3/dense_17/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/cnn__gender_3/dense_17/kernel/v*
_output_shapes

:*
dtype0
?
"Adam/cnn__gender_3/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/cnn__gender_3/dense_17/bias/v
?
6Adam/cnn__gender_3/dense_17/bias/v/Read/ReadVariableOpReadVariableOp"Adam/cnn__gender_3/dense_17/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?t
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?s
value?sB?s B?s
?
	conv1

batchnorm1
maxpool1
dropout1
	conv2

batchnorm2
maxpool2
dropout2
		conv3


batchnorm3
maxpool3
dropout3
flatten

dense1

dense2

dense3
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?
axis
	gamma
beta
 moving_mean
!moving_variance
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
R
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
?
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9regularization_losses
:	variables
;trainable_variables
<	keras_api
R
=regularization_losses
>	variables
?trainable_variables
@	keras_api
R
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
R
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
R
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
h

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
h

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
h

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
?
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem?m?m?m?.m?/m?5m?6m?Em?Fm?Lm?Mm?`m?am?fm?gm?lm?mm?v?v?v?v?.v?/v?5v?6v?Ev?Fv?Lv?Mv?`v?av?fv?gv?lv?mv?
 
?
0
1
2
3
 4
!5
.6
/7
58
69
710
811
E12
F13
L14
M15
N16
O17
`18
a19
f20
g21
l22
m23
?
0
1
2
3
.4
/5
56
67
E8
F9
L10
M11
`12
a13
f14
g15
l16
m17
?
wlayer_regularization_losses
regularization_losses
	variables

xlayers
trainable_variables
ylayer_metrics
znon_trainable_variables
{metrics
 
[Y
VARIABLE_VALUEcnn__gender_3/conv2d_15/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcnn__gender_3/conv2d_15/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
|layer_regularization_losses
regularization_losses
	variables

}layers
trainable_variables
~layer_metrics
non_trainable_variables
?metrics
 
ki
VARIABLE_VALUE*cnn__gender_3/batch_normalization_15/gamma+batchnorm1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE)cnn__gender_3/batch_normalization_15/beta*batchnorm1/beta/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0cnn__gender_3/batch_normalization_15/moving_mean1batchnorm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE4cnn__gender_3/batch_normalization_15/moving_variance5batchnorm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 2
!3

0
1
?
 ?layer_regularization_losses
"regularization_losses
#	variables
?layers
$trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
&regularization_losses
'	variables
?layers
(trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
*regularization_losses
+	variables
?layers
,trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
[Y
VARIABLE_VALUEcnn__gender_3/conv2d_16/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcnn__gender_3/conv2d_16/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
 ?layer_regularization_losses
0regularization_losses
1	variables
?layers
2trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
ki
VARIABLE_VALUE*cnn__gender_3/batch_normalization_16/gamma+batchnorm2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE)cnn__gender_3/batch_normalization_16/beta*batchnorm2/beta/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0cnn__gender_3/batch_normalization_16/moving_mean1batchnorm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE4cnn__gender_3/batch_normalization_16/moving_variance5batchnorm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

50
61
72
83

50
61
?
 ?layer_regularization_losses
9regularization_losses
:	variables
?layers
;trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
=regularization_losses
>	variables
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
Aregularization_losses
B	variables
?layers
Ctrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
[Y
VARIABLE_VALUEcnn__gender_3/conv2d_17/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcnn__gender_3/conv2d_17/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
?
 ?layer_regularization_losses
Gregularization_losses
H	variables
?layers
Itrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
ki
VARIABLE_VALUE*cnn__gender_3/batch_normalization_17/gamma+batchnorm3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE)cnn__gender_3/batch_normalization_17/beta*batchnorm3/beta/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0cnn__gender_3/batch_normalization_17/moving_mean1batchnorm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE4cnn__gender_3/batch_normalization_17/moving_variance5batchnorm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1
N2
O3

L0
M1
?
 ?layer_regularization_losses
Pregularization_losses
Q	variables
?layers
Rtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
Tregularization_losses
U	variables
?layers
Vtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
Xregularization_losses
Y	variables
?layers
Ztrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
 
 
 
?
 ?layer_regularization_losses
\regularization_losses
]	variables
?layers
^trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
[Y
VARIABLE_VALUEcnn__gender_3/dense_15/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcnn__gender_3/dense_15/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1

`0
a1
?
 ?layer_regularization_losses
bregularization_losses
c	variables
?layers
dtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
[Y
VARIABLE_VALUEcnn__gender_3/dense_16/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcnn__gender_3/dense_16/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

f0
g1
?
 ?layer_regularization_losses
hregularization_losses
i	variables
?layers
jtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
[Y
VARIABLE_VALUEcnn__gender_3/dense_17/kernel(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEcnn__gender_3/dense_17/bias&dense3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

l0
m1
?
 ?layer_regularization_losses
nregularization_losses
o	variables
?layers
ptrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
*
 0
!1
72
83
N4
O5

?0
?1
 
 
 
 
 
 
 
 

 0
!1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

N0
O1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUE%Adam/cnn__gender_3/conv2d_15/kernel/mCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/cnn__gender_3/conv2d_15/bias/mAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/cnn__gender_3/batch_normalization_15/gamma/mGbatchnorm1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/cnn__gender_3/batch_normalization_15/beta/mFbatchnorm1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE%Adam/cnn__gender_3/conv2d_16/kernel/mCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/cnn__gender_3/conv2d_16/bias/mAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/cnn__gender_3/batch_normalization_16/gamma/mGbatchnorm2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/cnn__gender_3/batch_normalization_16/beta/mFbatchnorm2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE%Adam/cnn__gender_3/conv2d_17/kernel/mCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/cnn__gender_3/conv2d_17/bias/mAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/cnn__gender_3/batch_normalization_17/gamma/mGbatchnorm3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/cnn__gender_3/batch_normalization_17/beta/mFbatchnorm3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/cnn__gender_3/dense_15/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/cnn__gender_3/dense_15/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/cnn__gender_3/dense_16/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/cnn__gender_3/dense_16/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/cnn__gender_3/dense_17/kernel/mDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/cnn__gender_3/dense_17/bias/mBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE%Adam/cnn__gender_3/conv2d_15/kernel/vCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/cnn__gender_3/conv2d_15/bias/vAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/cnn__gender_3/batch_normalization_15/gamma/vGbatchnorm1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/cnn__gender_3/batch_normalization_15/beta/vFbatchnorm1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE%Adam/cnn__gender_3/conv2d_16/kernel/vCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/cnn__gender_3/conv2d_16/bias/vAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/cnn__gender_3/batch_normalization_16/gamma/vGbatchnorm2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/cnn__gender_3/batch_normalization_16/beta/vFbatchnorm2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE%Adam/cnn__gender_3/conv2d_17/kernel/vCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE#Adam/cnn__gender_3/conv2d_17/bias/vAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/cnn__gender_3/batch_normalization_17/gamma/vGbatchnorm3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/cnn__gender_3/batch_normalization_17/beta/vFbatchnorm3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/cnn__gender_3/dense_15/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/cnn__gender_3/dense_15/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/cnn__gender_3/dense_16/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/cnn__gender_3/dense_16/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE$Adam/cnn__gender_3/dense_17/kernel/vDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/cnn__gender_3/dense_17/bias/vBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cnn__gender_3/conv2d_15/kernelcnn__gender_3/conv2d_15/bias*cnn__gender_3/batch_normalization_15/gamma)cnn__gender_3/batch_normalization_15/beta0cnn__gender_3/batch_normalization_15/moving_mean4cnn__gender_3/batch_normalization_15/moving_variancecnn__gender_3/conv2d_16/kernelcnn__gender_3/conv2d_16/bias*cnn__gender_3/batch_normalization_16/gamma)cnn__gender_3/batch_normalization_16/beta0cnn__gender_3/batch_normalization_16/moving_mean4cnn__gender_3/batch_normalization_16/moving_variancecnn__gender_3/conv2d_17/kernelcnn__gender_3/conv2d_17/bias*cnn__gender_3/batch_normalization_17/gamma)cnn__gender_3/batch_normalization_17/beta0cnn__gender_3/batch_normalization_17/moving_mean4cnn__gender_3/batch_normalization_17/moving_variancecnn__gender_3/dense_15/kernelcnn__gender_3/dense_15/biascnn__gender_3/dense_16/kernelcnn__gender_3/dense_16/biascnn__gender_3/dense_17/kernelcnn__gender_3/dense_17/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_17985
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2cnn__gender_3/conv2d_15/kernel/Read/ReadVariableOp0cnn__gender_3/conv2d_15/bias/Read/ReadVariableOp>cnn__gender_3/batch_normalization_15/gamma/Read/ReadVariableOp=cnn__gender_3/batch_normalization_15/beta/Read/ReadVariableOpDcnn__gender_3/batch_normalization_15/moving_mean/Read/ReadVariableOpHcnn__gender_3/batch_normalization_15/moving_variance/Read/ReadVariableOp2cnn__gender_3/conv2d_16/kernel/Read/ReadVariableOp0cnn__gender_3/conv2d_16/bias/Read/ReadVariableOp>cnn__gender_3/batch_normalization_16/gamma/Read/ReadVariableOp=cnn__gender_3/batch_normalization_16/beta/Read/ReadVariableOpDcnn__gender_3/batch_normalization_16/moving_mean/Read/ReadVariableOpHcnn__gender_3/batch_normalization_16/moving_variance/Read/ReadVariableOp2cnn__gender_3/conv2d_17/kernel/Read/ReadVariableOp0cnn__gender_3/conv2d_17/bias/Read/ReadVariableOp>cnn__gender_3/batch_normalization_17/gamma/Read/ReadVariableOp=cnn__gender_3/batch_normalization_17/beta/Read/ReadVariableOpDcnn__gender_3/batch_normalization_17/moving_mean/Read/ReadVariableOpHcnn__gender_3/batch_normalization_17/moving_variance/Read/ReadVariableOp1cnn__gender_3/dense_15/kernel/Read/ReadVariableOp/cnn__gender_3/dense_15/bias/Read/ReadVariableOp1cnn__gender_3/dense_16/kernel/Read/ReadVariableOp/cnn__gender_3/dense_16/bias/Read/ReadVariableOp1cnn__gender_3/dense_17/kernel/Read/ReadVariableOp/cnn__gender_3/dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp9Adam/cnn__gender_3/conv2d_15/kernel/m/Read/ReadVariableOp7Adam/cnn__gender_3/conv2d_15/bias/m/Read/ReadVariableOpEAdam/cnn__gender_3/batch_normalization_15/gamma/m/Read/ReadVariableOpDAdam/cnn__gender_3/batch_normalization_15/beta/m/Read/ReadVariableOp9Adam/cnn__gender_3/conv2d_16/kernel/m/Read/ReadVariableOp7Adam/cnn__gender_3/conv2d_16/bias/m/Read/ReadVariableOpEAdam/cnn__gender_3/batch_normalization_16/gamma/m/Read/ReadVariableOpDAdam/cnn__gender_3/batch_normalization_16/beta/m/Read/ReadVariableOp9Adam/cnn__gender_3/conv2d_17/kernel/m/Read/ReadVariableOp7Adam/cnn__gender_3/conv2d_17/bias/m/Read/ReadVariableOpEAdam/cnn__gender_3/batch_normalization_17/gamma/m/Read/ReadVariableOpDAdam/cnn__gender_3/batch_normalization_17/beta/m/Read/ReadVariableOp8Adam/cnn__gender_3/dense_15/kernel/m/Read/ReadVariableOp6Adam/cnn__gender_3/dense_15/bias/m/Read/ReadVariableOp8Adam/cnn__gender_3/dense_16/kernel/m/Read/ReadVariableOp6Adam/cnn__gender_3/dense_16/bias/m/Read/ReadVariableOp8Adam/cnn__gender_3/dense_17/kernel/m/Read/ReadVariableOp6Adam/cnn__gender_3/dense_17/bias/m/Read/ReadVariableOp9Adam/cnn__gender_3/conv2d_15/kernel/v/Read/ReadVariableOp7Adam/cnn__gender_3/conv2d_15/bias/v/Read/ReadVariableOpEAdam/cnn__gender_3/batch_normalization_15/gamma/v/Read/ReadVariableOpDAdam/cnn__gender_3/batch_normalization_15/beta/v/Read/ReadVariableOp9Adam/cnn__gender_3/conv2d_16/kernel/v/Read/ReadVariableOp7Adam/cnn__gender_3/conv2d_16/bias/v/Read/ReadVariableOpEAdam/cnn__gender_3/batch_normalization_16/gamma/v/Read/ReadVariableOpDAdam/cnn__gender_3/batch_normalization_16/beta/v/Read/ReadVariableOp9Adam/cnn__gender_3/conv2d_17/kernel/v/Read/ReadVariableOp7Adam/cnn__gender_3/conv2d_17/bias/v/Read/ReadVariableOpEAdam/cnn__gender_3/batch_normalization_17/gamma/v/Read/ReadVariableOpDAdam/cnn__gender_3/batch_normalization_17/beta/v/Read/ReadVariableOp8Adam/cnn__gender_3/dense_15/kernel/v/Read/ReadVariableOp6Adam/cnn__gender_3/dense_15/bias/v/Read/ReadVariableOp8Adam/cnn__gender_3/dense_16/kernel/v/Read/ReadVariableOp6Adam/cnn__gender_3/dense_16/bias/v/Read/ReadVariableOp8Adam/cnn__gender_3/dense_17/kernel/v/Read/ReadVariableOp6Adam/cnn__gender_3/dense_17/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_19437
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn__gender_3/conv2d_15/kernelcnn__gender_3/conv2d_15/bias*cnn__gender_3/batch_normalization_15/gamma)cnn__gender_3/batch_normalization_15/beta0cnn__gender_3/batch_normalization_15/moving_mean4cnn__gender_3/batch_normalization_15/moving_variancecnn__gender_3/conv2d_16/kernelcnn__gender_3/conv2d_16/bias*cnn__gender_3/batch_normalization_16/gamma)cnn__gender_3/batch_normalization_16/beta0cnn__gender_3/batch_normalization_16/moving_mean4cnn__gender_3/batch_normalization_16/moving_variancecnn__gender_3/conv2d_17/kernelcnn__gender_3/conv2d_17/bias*cnn__gender_3/batch_normalization_17/gamma)cnn__gender_3/batch_normalization_17/beta0cnn__gender_3/batch_normalization_17/moving_mean4cnn__gender_3/batch_normalization_17/moving_variancecnn__gender_3/dense_15/kernelcnn__gender_3/dense_15/biascnn__gender_3/dense_16/kernelcnn__gender_3/dense_16/biascnn__gender_3/dense_17/kernelcnn__gender_3/dense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1%Adam/cnn__gender_3/conv2d_15/kernel/m#Adam/cnn__gender_3/conv2d_15/bias/m1Adam/cnn__gender_3/batch_normalization_15/gamma/m0Adam/cnn__gender_3/batch_normalization_15/beta/m%Adam/cnn__gender_3/conv2d_16/kernel/m#Adam/cnn__gender_3/conv2d_16/bias/m1Adam/cnn__gender_3/batch_normalization_16/gamma/m0Adam/cnn__gender_3/batch_normalization_16/beta/m%Adam/cnn__gender_3/conv2d_17/kernel/m#Adam/cnn__gender_3/conv2d_17/bias/m1Adam/cnn__gender_3/batch_normalization_17/gamma/m0Adam/cnn__gender_3/batch_normalization_17/beta/m$Adam/cnn__gender_3/dense_15/kernel/m"Adam/cnn__gender_3/dense_15/bias/m$Adam/cnn__gender_3/dense_16/kernel/m"Adam/cnn__gender_3/dense_16/bias/m$Adam/cnn__gender_3/dense_17/kernel/m"Adam/cnn__gender_3/dense_17/bias/m%Adam/cnn__gender_3/conv2d_15/kernel/v#Adam/cnn__gender_3/conv2d_15/bias/v1Adam/cnn__gender_3/batch_normalization_15/gamma/v0Adam/cnn__gender_3/batch_normalization_15/beta/v%Adam/cnn__gender_3/conv2d_16/kernel/v#Adam/cnn__gender_3/conv2d_16/bias/v1Adam/cnn__gender_3/batch_normalization_16/gamma/v0Adam/cnn__gender_3/batch_normalization_16/beta/v%Adam/cnn__gender_3/conv2d_17/kernel/v#Adam/cnn__gender_3/conv2d_17/bias/v1Adam/cnn__gender_3/batch_normalization_17/gamma/v0Adam/cnn__gender_3/batch_normalization_17/beta/v$Adam/cnn__gender_3/dense_15/kernel/v"Adam/cnn__gender_3/dense_15/bias/v$Adam/cnn__gender_3/dense_16/kernel/v"Adam/cnn__gender_3/dense_16/bias/v$Adam/cnn__gender_3/dense_17/kernel/v"Adam/cnn__gender_3/dense_17/bias/v*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_19654??
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19003

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_15_layer_call_and_return_conditional_losses_17202

inputs2
matmul_readvariableop_resource:
?? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_cnn__gender_3_layer_call_fn_18464
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:
?? 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_172432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_17038

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18850

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?N
?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_17243

inputs)
conv2d_15_17039: 
conv2d_15_17041: *
batch_normalization_15_17062: *
batch_normalization_15_17064: *
batch_normalization_15_17066: *
batch_normalization_15_17068: )
conv2d_16_17091: @
conv2d_16_17093:@*
batch_normalization_16_17114:@*
batch_normalization_16_17116:@*
batch_normalization_16_17118:@*
batch_normalization_16_17120:@*
conv2d_17_17143:@?
conv2d_17_17145:	?+
batch_normalization_17_17166:	?+
batch_normalization_17_17168:	?+
batch_normalization_17_17170:	?+
batch_normalization_17_17172:	?"
dense_15_17203:
?? 
dense_15_17205:  
dense_16_17220: 
dense_16_17222: 
dense_17_17237:
dense_17_17239:
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_17039conv2d_15_17041*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_170382#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_15_17062batch_normalization_15_17064batch_normalization_15_17066batch_normalization_15_17068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1706120
.batch_normalization_15/StatefulPartitionedCall?
 max_pooling2d_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_167382"
 max_pooling2d_15/PartitionedCall?
dropout_15/PartitionedCallPartitionedCall)max_pooling2d_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_170772
dropout_15/PartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0conv2d_16_17091conv2d_16_17093*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_170902#
!conv2d_16/StatefulPartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_16_17114batch_normalization_16_17116batch_normalization_16_17118batch_normalization_16_17120*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1711320
.batch_normalization_16/StatefulPartitionedCall?
 max_pooling2d_16/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_168762"
 max_pooling2d_16/PartitionedCall?
dropout_16/PartitionedCallPartitionedCall)max_pooling2d_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_171292
dropout_16/PartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0conv2d_17_17143conv2d_17_17145*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????..?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_171422#
!conv2d_17/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_17_17166batch_normalization_17_17168batch_normalization_17_17170batch_normalization_17_17172*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????..?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1716520
.batch_normalization_17/StatefulPartitionedCall?
 max_pooling2d_17/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_170142"
 max_pooling2d_17/PartitionedCall?
dropout_17/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_171812
dropout_17/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall#dropout_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_171892
flatten_5/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_17203dense_15_17205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_172022"
 dense_15/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_17220dense_16_17222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_172192"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_17237dense_17_17239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_172362"
 dense_17/StatefulPartitionedCall?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_16_layer_call_fn_16882

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_168762
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?S
?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_17684

inputs)
conv2d_15_17619: 
conv2d_15_17621: *
batch_normalization_15_17624: *
batch_normalization_15_17626: *
batch_normalization_15_17628: *
batch_normalization_15_17630: )
conv2d_16_17635: @
conv2d_16_17637:@*
batch_normalization_16_17640:@*
batch_normalization_16_17642:@*
batch_normalization_16_17644:@*
batch_normalization_16_17646:@*
conv2d_17_17651:@?
conv2d_17_17653:	?+
batch_normalization_17_17656:	?+
batch_normalization_17_17658:	?+
batch_normalization_17_17660:	?+
batch_normalization_17_17662:	?"
dense_15_17668:
?? 
dense_15_17670:  
dense_16_17673: 
dense_16_17675: 
dense_17_17678:
dense_17_17680:
identity??.batch_normalization_15/StatefulPartitionedCall?.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?"dropout_15/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_17619conv2d_15_17621*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_170382#
!conv2d_15/StatefulPartitionedCall?
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_15_17624batch_normalization_15_17626batch_normalization_15_17628batch_normalization_15_17630*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_1754020
.batch_normalization_15/StatefulPartitionedCall?
 max_pooling2d_15/PartitionedCallPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_167382"
 max_pooling2d_15/PartitionedCall?
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_175042$
"dropout_15/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0conv2d_16_17635conv2d_16_17637*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_170902#
!conv2d_16/StatefulPartitionedCall?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0batch_normalization_16_17640batch_normalization_16_17642batch_normalization_16_17644batch_normalization_16_17646*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1746320
.batch_normalization_16/StatefulPartitionedCall?
 max_pooling2d_16/PartitionedCallPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_168762"
 max_pooling2d_16/PartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_174272$
"dropout_16/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0conv2d_17_17651conv2d_17_17653*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????..?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_171422#
!conv2d_17/StatefulPartitionedCall?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0batch_normalization_17_17656batch_normalization_17_17658batch_normalization_17_17660batch_normalization_17_17662*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????..?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1738620
.batch_normalization_17/StatefulPartitionedCall?
 max_pooling2d_17/PartitionedCallPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_170142"
 max_pooling2d_17/PartitionedCall?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_17/PartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_173502$
"dropout_17/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCall+dropout_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_171892
flatten_5/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_17668dense_15_17670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_172022"
 dense_15/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_17673dense_16_17675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_172192"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_17678dense_17_17680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_172362"
 dense_17/StatefulPartitionedCall?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_dense_17_layer_call_and_return_conditional_losses_19198

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18081

inputsB
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: <
.batch_normalization_15_readvariableop_resource: >
0batch_normalization_15_readvariableop_1_resource: M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@<
.batch_normalization_16_readvariableop_resource:@>
0batch_normalization_16_readvariableop_1_resource:@M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_17_conv2d_readvariableop_resource:@?8
)conv2d_17_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_15_matmul_readvariableop_resource:
?? 6
(dense_15_biasadd_readvariableop_resource: 9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:6
(dense_17_biasadd_readvariableop_resource:
identity??6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/BiasAdd?
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/Relu?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3?
max_pooling2d_15/MaxPoolMaxPool+batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????cc *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool?
dropout_15/IdentityIdentity!max_pooling2d_15/MaxPool:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout_15/Identity?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2DConv2Ddropout_15/Identity:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@*
paddingVALID*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/Relu?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3?
max_pooling2d_16/MaxPoolMaxPool+batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool?
dropout_16/IdentityIdentity!max_pooling2d_16/MaxPool:output:0*
T0*/
_output_shapes
:?????????00@2
dropout_16/Identity?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2DConv2Ddropout_16/Identity:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?*
paddingVALID*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/Relu?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_17/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3?
max_pooling2d_17/MaxPoolMaxPool+batch_normalization_17/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool?
dropout_17/IdentityIdentity!max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_17/Identitys
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_17/Identity:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshape?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_15/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Sigmoid?
IdentityIdentitydense_17/Sigmoid:y:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17165

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????..?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????..?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_16628

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_18634

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_16_layer_call_fn_18965

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_174272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_18741

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_166722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
-__inference_cnn__gender_3_layer_call_fn_18517

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:
?? 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_172432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_17_layer_call_and_return_conditional_losses_19126

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_16948

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18198

inputsB
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: <
.batch_normalization_15_readvariableop_resource: >
0batch_normalization_15_readvariableop_1_resource: M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@<
.batch_normalization_16_readvariableop_resource:@>
0batch_normalization_16_readvariableop_1_resource:@M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_17_conv2d_readvariableop_resource:@?8
)conv2d_17_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_15_matmul_readvariableop_resource:
?? 6
(dense_15_biasadd_readvariableop_resource: 9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:6
(dense_17_biasadd_readvariableop_resource:
identity??%batch_normalization_15/AssignNewValue?'batch_normalization_15/AssignNewValue_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?%batch_normalization_16/AssignNewValue?'batch_normalization_16/AssignNewValue_1?6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?%batch_normalization_17/AssignNewValue?'batch_normalization_17/AssignNewValue_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/BiasAdd?
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/Relu?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_15/FusedBatchNormV3?
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValue?
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1?
max_pooling2d_15/MaxPoolMaxPool+batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????cc *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPooly
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_15/dropout/Const?
dropout_15/dropout/MulMul!max_pooling2d_15/MaxPool:output:0!dropout_15/dropout/Const:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout_15/dropout/Mul?
dropout_15/dropout/ShapeShape!max_pooling2d_15/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape?
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????cc *
dtype021
/dropout_15/dropout/random_uniform/RandomUniform?
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_15/dropout/GreaterEqual/y?
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????cc 2!
dropout_15/dropout/GreaterEqual?
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????cc 2
dropout_15/dropout/Cast?
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????cc 2
dropout_15/dropout/Mul_1?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2DConv2Ddropout_15/dropout/Mul_1:z:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@*
paddingVALID*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/Relu?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_16/FusedBatchNormV3?
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue?
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1?
max_pooling2d_16/MaxPoolMaxPool+batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPooly
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_16/dropout/Const?
dropout_16/dropout/MulMul!max_pooling2d_16/MaxPool:output:0!dropout_16/dropout/Const:output:0*
T0*/
_output_shapes
:?????????00@2
dropout_16/dropout/Mul?
dropout_16/dropout/ShapeShape!max_pooling2d_16/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????00@*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform?
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_16/dropout/GreaterEqual/y?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????00@2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????00@2
dropout_16/dropout/Cast?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????00@2
dropout_16/dropout/Mul_1?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2DConv2Ddropout_16/dropout/Mul_1:z:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?*
paddingVALID*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/Relu?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_17/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_17/FusedBatchNormV3?
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue?
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1?
max_pooling2d_17/MaxPoolMaxPool+batch_normalization_17/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPooly
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMul!max_pooling2d_17/MaxPool:output:0!dropout_17/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_17/dropout/Mul?
dropout_17/dropout/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_17/dropout/Mul_1s
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_17/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshape?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_15/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Sigmoid?

IdentityIdentitydense_17/Sigmoid:y:0&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18294
input_1B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: <
.batch_normalization_15_readvariableop_resource: >
0batch_normalization_15_readvariableop_1_resource: M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@<
.batch_normalization_16_readvariableop_resource:@>
0batch_normalization_16_readvariableop_1_resource:@M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_17_conv2d_readvariableop_resource:@?8
)conv2d_17_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_15_matmul_readvariableop_resource:
?? 6
(dense_15_biasadd_readvariableop_resource: 9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:6
(dense_17_biasadd_readvariableop_resource:
identity??6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dinput_1'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/BiasAdd?
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/Relu?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3?
max_pooling2d_15/MaxPoolMaxPool+batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????cc *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool?
dropout_15/IdentityIdentity!max_pooling2d_15/MaxPool:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout_15/Identity?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2DConv2Ddropout_15/Identity:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@*
paddingVALID*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/Relu?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3?
max_pooling2d_16/MaxPoolMaxPool+batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool?
dropout_16/IdentityIdentity!max_pooling2d_16/MaxPool:output:0*
T0*/
_output_shapes
:?????????00@2
dropout_16/Identity?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2DConv2Ddropout_16/Identity:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?*
paddingVALID*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/Relu?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_17/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3?
max_pooling2d_17/MaxPoolMaxPool+batch_normalization_17/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool?
dropout_17/IdentityIdentity!max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_17/Identitys
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_17/Identity:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshape?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_15/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Sigmoid?
IdentityIdentitydense_17/Sigmoid:y:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
d
E__inference_dropout_17_layer_call_and_return_conditional_losses_17350

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_16810

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_19070

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_169042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18868

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????aa@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????aa@
 
_user_specified_nameinputs
?
d
E__inference_dropout_15_layer_call_and_return_conditional_losses_18784

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????cc *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????cc 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????cc 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????cc 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cc :W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?
?
(__inference_dense_17_layer_call_fn_19207

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_172362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_18976

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????..?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_18728

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_166282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_16904

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_16_layer_call_and_return_conditional_losses_18955

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????00@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????00@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????00@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????00@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????00@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00@:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_17985
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:
?? 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_166062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
F
*__inference_dropout_15_layer_call_fn_18789

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_170772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????cc 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cc :W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?
?
(__inference_dense_15_layer_call_fn_19167

inputs
unknown:
?? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_172022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18697

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_16672

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_19096

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????..?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_171652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????..?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????..?
 
_user_specified_nameinputs
?
?
)__inference_conv2d_16_layer_call_fn_18814

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_16_layer_call_and_return_conditional_losses_170902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????cc : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_17014

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17540

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_18754

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_170612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_15_layer_call_fn_16744

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_167382
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18661

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_15_layer_call_fn_18643

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_15_layer_call_and_return_conditional_losses_170382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_16_layer_call_fn_18960

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_171292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00@:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_19142

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_15_layer_call_and_return_conditional_losses_19158

inputs2
matmul_readvariableop_resource:
?? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_5_layer_call_fn_19147

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_171892
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17386

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????..?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????..?
 
_user_specified_nameinputs
?
?
(__inference_dense_16_layer_call_fn_19187

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_172192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_17_layer_call_fn_19136

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_173502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_17090

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????aa@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????cc : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?
c
E__inference_dropout_15_layer_call_and_return_conditional_losses_18772

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????cc 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????cc 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cc :W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?

?
C__inference_dense_16_layer_call_and_return_conditional_losses_19178

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_18943

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????00@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????00@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00@:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_16_layer_call_fn_18899

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_167662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19057

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????..?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????..?
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_19109

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????..?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_173862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????..?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????..?
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_17_layer_call_fn_17020

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_170142
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?%
__inference__traced_save_19437
file_prefix=
9savev2_cnn__gender_3_conv2d_15_kernel_read_readvariableop;
7savev2_cnn__gender_3_conv2d_15_bias_read_readvariableopI
Esavev2_cnn__gender_3_batch_normalization_15_gamma_read_readvariableopH
Dsavev2_cnn__gender_3_batch_normalization_15_beta_read_readvariableopO
Ksavev2_cnn__gender_3_batch_normalization_15_moving_mean_read_readvariableopS
Osavev2_cnn__gender_3_batch_normalization_15_moving_variance_read_readvariableop=
9savev2_cnn__gender_3_conv2d_16_kernel_read_readvariableop;
7savev2_cnn__gender_3_conv2d_16_bias_read_readvariableopI
Esavev2_cnn__gender_3_batch_normalization_16_gamma_read_readvariableopH
Dsavev2_cnn__gender_3_batch_normalization_16_beta_read_readvariableopO
Ksavev2_cnn__gender_3_batch_normalization_16_moving_mean_read_readvariableopS
Osavev2_cnn__gender_3_batch_normalization_16_moving_variance_read_readvariableop=
9savev2_cnn__gender_3_conv2d_17_kernel_read_readvariableop;
7savev2_cnn__gender_3_conv2d_17_bias_read_readvariableopI
Esavev2_cnn__gender_3_batch_normalization_17_gamma_read_readvariableopH
Dsavev2_cnn__gender_3_batch_normalization_17_beta_read_readvariableopO
Ksavev2_cnn__gender_3_batch_normalization_17_moving_mean_read_readvariableopS
Osavev2_cnn__gender_3_batch_normalization_17_moving_variance_read_readvariableop<
8savev2_cnn__gender_3_dense_15_kernel_read_readvariableop:
6savev2_cnn__gender_3_dense_15_bias_read_readvariableop<
8savev2_cnn__gender_3_dense_16_kernel_read_readvariableop:
6savev2_cnn__gender_3_dense_16_bias_read_readvariableop<
8savev2_cnn__gender_3_dense_17_kernel_read_readvariableop:
6savev2_cnn__gender_3_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopD
@savev2_adam_cnn__gender_3_conv2d_15_kernel_m_read_readvariableopB
>savev2_adam_cnn__gender_3_conv2d_15_bias_m_read_readvariableopP
Lsavev2_adam_cnn__gender_3_batch_normalization_15_gamma_m_read_readvariableopO
Ksavev2_adam_cnn__gender_3_batch_normalization_15_beta_m_read_readvariableopD
@savev2_adam_cnn__gender_3_conv2d_16_kernel_m_read_readvariableopB
>savev2_adam_cnn__gender_3_conv2d_16_bias_m_read_readvariableopP
Lsavev2_adam_cnn__gender_3_batch_normalization_16_gamma_m_read_readvariableopO
Ksavev2_adam_cnn__gender_3_batch_normalization_16_beta_m_read_readvariableopD
@savev2_adam_cnn__gender_3_conv2d_17_kernel_m_read_readvariableopB
>savev2_adam_cnn__gender_3_conv2d_17_bias_m_read_readvariableopP
Lsavev2_adam_cnn__gender_3_batch_normalization_17_gamma_m_read_readvariableopO
Ksavev2_adam_cnn__gender_3_batch_normalization_17_beta_m_read_readvariableopC
?savev2_adam_cnn__gender_3_dense_15_kernel_m_read_readvariableopA
=savev2_adam_cnn__gender_3_dense_15_bias_m_read_readvariableopC
?savev2_adam_cnn__gender_3_dense_16_kernel_m_read_readvariableopA
=savev2_adam_cnn__gender_3_dense_16_bias_m_read_readvariableopC
?savev2_adam_cnn__gender_3_dense_17_kernel_m_read_readvariableopA
=savev2_adam_cnn__gender_3_dense_17_bias_m_read_readvariableopD
@savev2_adam_cnn__gender_3_conv2d_15_kernel_v_read_readvariableopB
>savev2_adam_cnn__gender_3_conv2d_15_bias_v_read_readvariableopP
Lsavev2_adam_cnn__gender_3_batch_normalization_15_gamma_v_read_readvariableopO
Ksavev2_adam_cnn__gender_3_batch_normalization_15_beta_v_read_readvariableopD
@savev2_adam_cnn__gender_3_conv2d_16_kernel_v_read_readvariableopB
>savev2_adam_cnn__gender_3_conv2d_16_bias_v_read_readvariableopP
Lsavev2_adam_cnn__gender_3_batch_normalization_16_gamma_v_read_readvariableopO
Ksavev2_adam_cnn__gender_3_batch_normalization_16_beta_v_read_readvariableopD
@savev2_adam_cnn__gender_3_conv2d_17_kernel_v_read_readvariableopB
>savev2_adam_cnn__gender_3_conv2d_17_bias_v_read_readvariableopP
Lsavev2_adam_cnn__gender_3_batch_normalization_17_gamma_v_read_readvariableopO
Ksavev2_adam_cnn__gender_3_batch_normalization_17_beta_v_read_readvariableopC
?savev2_adam_cnn__gender_3_dense_15_kernel_v_read_readvariableopA
=savev2_adam_cnn__gender_3_dense_15_bias_v_read_readvariableopC
?savev2_adam_cnn__gender_3_dense_16_kernel_v_read_readvariableopA
=savev2_adam_cnn__gender_3_dense_16_bias_v_read_readvariableopC
?savev2_adam_cnn__gender_3_dense_17_kernel_v_read_readvariableopA
=savev2_adam_cnn__gender_3_dense_17_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename? 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB+batchnorm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batchnorm1/beta/.ATTRIBUTES/VARIABLE_VALUEB1batchnorm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batchnorm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB+batchnorm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batchnorm2/beta/.ATTRIBUTES/VARIABLE_VALUEB1batchnorm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batchnorm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB+batchnorm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batchnorm3/beta/.ATTRIBUTES/VARIABLE_VALUEB1batchnorm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batchnorm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_cnn__gender_3_conv2d_15_kernel_read_readvariableop7savev2_cnn__gender_3_conv2d_15_bias_read_readvariableopEsavev2_cnn__gender_3_batch_normalization_15_gamma_read_readvariableopDsavev2_cnn__gender_3_batch_normalization_15_beta_read_readvariableopKsavev2_cnn__gender_3_batch_normalization_15_moving_mean_read_readvariableopOsavev2_cnn__gender_3_batch_normalization_15_moving_variance_read_readvariableop9savev2_cnn__gender_3_conv2d_16_kernel_read_readvariableop7savev2_cnn__gender_3_conv2d_16_bias_read_readvariableopEsavev2_cnn__gender_3_batch_normalization_16_gamma_read_readvariableopDsavev2_cnn__gender_3_batch_normalization_16_beta_read_readvariableopKsavev2_cnn__gender_3_batch_normalization_16_moving_mean_read_readvariableopOsavev2_cnn__gender_3_batch_normalization_16_moving_variance_read_readvariableop9savev2_cnn__gender_3_conv2d_17_kernel_read_readvariableop7savev2_cnn__gender_3_conv2d_17_bias_read_readvariableopEsavev2_cnn__gender_3_batch_normalization_17_gamma_read_readvariableopDsavev2_cnn__gender_3_batch_normalization_17_beta_read_readvariableopKsavev2_cnn__gender_3_batch_normalization_17_moving_mean_read_readvariableopOsavev2_cnn__gender_3_batch_normalization_17_moving_variance_read_readvariableop8savev2_cnn__gender_3_dense_15_kernel_read_readvariableop6savev2_cnn__gender_3_dense_15_bias_read_readvariableop8savev2_cnn__gender_3_dense_16_kernel_read_readvariableop6savev2_cnn__gender_3_dense_16_bias_read_readvariableop8savev2_cnn__gender_3_dense_17_kernel_read_readvariableop6savev2_cnn__gender_3_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop@savev2_adam_cnn__gender_3_conv2d_15_kernel_m_read_readvariableop>savev2_adam_cnn__gender_3_conv2d_15_bias_m_read_readvariableopLsavev2_adam_cnn__gender_3_batch_normalization_15_gamma_m_read_readvariableopKsavev2_adam_cnn__gender_3_batch_normalization_15_beta_m_read_readvariableop@savev2_adam_cnn__gender_3_conv2d_16_kernel_m_read_readvariableop>savev2_adam_cnn__gender_3_conv2d_16_bias_m_read_readvariableopLsavev2_adam_cnn__gender_3_batch_normalization_16_gamma_m_read_readvariableopKsavev2_adam_cnn__gender_3_batch_normalization_16_beta_m_read_readvariableop@savev2_adam_cnn__gender_3_conv2d_17_kernel_m_read_readvariableop>savev2_adam_cnn__gender_3_conv2d_17_bias_m_read_readvariableopLsavev2_adam_cnn__gender_3_batch_normalization_17_gamma_m_read_readvariableopKsavev2_adam_cnn__gender_3_batch_normalization_17_beta_m_read_readvariableop?savev2_adam_cnn__gender_3_dense_15_kernel_m_read_readvariableop=savev2_adam_cnn__gender_3_dense_15_bias_m_read_readvariableop?savev2_adam_cnn__gender_3_dense_16_kernel_m_read_readvariableop=savev2_adam_cnn__gender_3_dense_16_bias_m_read_readvariableop?savev2_adam_cnn__gender_3_dense_17_kernel_m_read_readvariableop=savev2_adam_cnn__gender_3_dense_17_bias_m_read_readvariableop@savev2_adam_cnn__gender_3_conv2d_15_kernel_v_read_readvariableop>savev2_adam_cnn__gender_3_conv2d_15_bias_v_read_readvariableopLsavev2_adam_cnn__gender_3_batch_normalization_15_gamma_v_read_readvariableopKsavev2_adam_cnn__gender_3_batch_normalization_15_beta_v_read_readvariableop@savev2_adam_cnn__gender_3_conv2d_16_kernel_v_read_readvariableop>savev2_adam_cnn__gender_3_conv2d_16_bias_v_read_readvariableopLsavev2_adam_cnn__gender_3_batch_normalization_16_gamma_v_read_readvariableopKsavev2_adam_cnn__gender_3_batch_normalization_16_beta_v_read_readvariableop@savev2_adam_cnn__gender_3_conv2d_17_kernel_v_read_readvariableop>savev2_adam_cnn__gender_3_conv2d_17_bias_v_read_readvariableopLsavev2_adam_cnn__gender_3_batch_normalization_17_gamma_v_read_readvariableopKsavev2_adam_cnn__gender_3_batch_normalization_17_beta_v_read_readvariableop?savev2_adam_cnn__gender_3_dense_15_kernel_v_read_readvariableop=savev2_adam_cnn__gender_3_dense_15_bias_v_read_readvariableop?savev2_adam_cnn__gender_3_dense_16_kernel_v_read_readvariableop=savev2_adam_cnn__gender_3_dense_16_bias_v_read_readvariableop?savev2_adam_cnn__gender_3_dense_17_kernel_v_read_readvariableop=savev2_adam_cnn__gender_3_dense_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : @:@:@:@:@:@:@?:?:?:?:?:?:
?? : : :::: : : : : : : : : : : : : : @:@:@:@:@?:?:?:?:
?? : : :::: : : : : @:@:@:@:@?:?:?:?:
?? : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
?? : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:-*)
'
_output_shapes
:@?:!+

_output_shapes	
:?:!,

_output_shapes	
:?:!-

_output_shapes	
:?:&."
 
_output_shapes
:
?? : /

_output_shapes
: :$0 

_output_shapes

: : 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:-<)
'
_output_shapes
:@?:!=

_output_shapes	
:?:!>

_output_shapes	
:?:!?

_output_shapes	
:?:&@"
 
_output_shapes
:
?? : A

_output_shapes
: :$B 

_output_shapes

: : C

_output_shapes
::$D 

_output_shapes

:: E

_output_shapes
::F

_output_shapes
: 
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17463

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????aa@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????aa@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18679

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
F
*__inference_dropout_17_layer_call_fn_19131

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_171812
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_cnn__gender_3_layer_call_fn_18570

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:
?? 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_176842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_17142

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????..?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?

?
C__inference_dense_16_layer_call_and_return_conditional_losses_17219

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_16876

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18715

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_15_layer_call_fn_18794

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_15_layer_call_and_return_conditional_losses_175042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????cc 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cc 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?
d
E__inference_dropout_15_layer_call_and_return_conditional_losses_17504

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????cc *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????cc 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????cc 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????cc 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cc :W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
??
?3
!__inference__traced_restore_19654
file_prefixI
/assignvariableop_cnn__gender_3_conv2d_15_kernel: =
/assignvariableop_1_cnn__gender_3_conv2d_15_bias: K
=assignvariableop_2_cnn__gender_3_batch_normalization_15_gamma: J
<assignvariableop_3_cnn__gender_3_batch_normalization_15_beta: Q
Cassignvariableop_4_cnn__gender_3_batch_normalization_15_moving_mean: U
Gassignvariableop_5_cnn__gender_3_batch_normalization_15_moving_variance: K
1assignvariableop_6_cnn__gender_3_conv2d_16_kernel: @=
/assignvariableop_7_cnn__gender_3_conv2d_16_bias:@K
=assignvariableop_8_cnn__gender_3_batch_normalization_16_gamma:@J
<assignvariableop_9_cnn__gender_3_batch_normalization_16_beta:@R
Dassignvariableop_10_cnn__gender_3_batch_normalization_16_moving_mean:@V
Hassignvariableop_11_cnn__gender_3_batch_normalization_16_moving_variance:@M
2assignvariableop_12_cnn__gender_3_conv2d_17_kernel:@??
0assignvariableop_13_cnn__gender_3_conv2d_17_bias:	?M
>assignvariableop_14_cnn__gender_3_batch_normalization_17_gamma:	?L
=assignvariableop_15_cnn__gender_3_batch_normalization_17_beta:	?S
Dassignvariableop_16_cnn__gender_3_batch_normalization_17_moving_mean:	?W
Hassignvariableop_17_cnn__gender_3_batch_normalization_17_moving_variance:	?E
1assignvariableop_18_cnn__gender_3_dense_15_kernel:
?? =
/assignvariableop_19_cnn__gender_3_dense_15_bias: C
1assignvariableop_20_cnn__gender_3_dense_16_kernel: =
/assignvariableop_21_cnn__gender_3_dense_16_bias:C
1assignvariableop_22_cnn__gender_3_dense_17_kernel:=
/assignvariableop_23_cnn__gender_3_dense_17_bias:'
assignvariableop_24_adam_iter:	 )
assignvariableop_25_adam_beta_1: )
assignvariableop_26_adam_beta_2: (
assignvariableop_27_adam_decay: 0
&assignvariableop_28_adam_learning_rate: #
assignvariableop_29_total: #
assignvariableop_30_count: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: S
9assignvariableop_33_adam_cnn__gender_3_conv2d_15_kernel_m: E
7assignvariableop_34_adam_cnn__gender_3_conv2d_15_bias_m: S
Eassignvariableop_35_adam_cnn__gender_3_batch_normalization_15_gamma_m: R
Dassignvariableop_36_adam_cnn__gender_3_batch_normalization_15_beta_m: S
9assignvariableop_37_adam_cnn__gender_3_conv2d_16_kernel_m: @E
7assignvariableop_38_adam_cnn__gender_3_conv2d_16_bias_m:@S
Eassignvariableop_39_adam_cnn__gender_3_batch_normalization_16_gamma_m:@R
Dassignvariableop_40_adam_cnn__gender_3_batch_normalization_16_beta_m:@T
9assignvariableop_41_adam_cnn__gender_3_conv2d_17_kernel_m:@?F
7assignvariableop_42_adam_cnn__gender_3_conv2d_17_bias_m:	?T
Eassignvariableop_43_adam_cnn__gender_3_batch_normalization_17_gamma_m:	?S
Dassignvariableop_44_adam_cnn__gender_3_batch_normalization_17_beta_m:	?L
8assignvariableop_45_adam_cnn__gender_3_dense_15_kernel_m:
?? D
6assignvariableop_46_adam_cnn__gender_3_dense_15_bias_m: J
8assignvariableop_47_adam_cnn__gender_3_dense_16_kernel_m: D
6assignvariableop_48_adam_cnn__gender_3_dense_16_bias_m:J
8assignvariableop_49_adam_cnn__gender_3_dense_17_kernel_m:D
6assignvariableop_50_adam_cnn__gender_3_dense_17_bias_m:S
9assignvariableop_51_adam_cnn__gender_3_conv2d_15_kernel_v: E
7assignvariableop_52_adam_cnn__gender_3_conv2d_15_bias_v: S
Eassignvariableop_53_adam_cnn__gender_3_batch_normalization_15_gamma_v: R
Dassignvariableop_54_adam_cnn__gender_3_batch_normalization_15_beta_v: S
9assignvariableop_55_adam_cnn__gender_3_conv2d_16_kernel_v: @E
7assignvariableop_56_adam_cnn__gender_3_conv2d_16_bias_v:@S
Eassignvariableop_57_adam_cnn__gender_3_batch_normalization_16_gamma_v:@R
Dassignvariableop_58_adam_cnn__gender_3_batch_normalization_16_beta_v:@T
9assignvariableop_59_adam_cnn__gender_3_conv2d_17_kernel_v:@?F
7assignvariableop_60_adam_cnn__gender_3_conv2d_17_bias_v:	?T
Eassignvariableop_61_adam_cnn__gender_3_batch_normalization_17_gamma_v:	?S
Dassignvariableop_62_adam_cnn__gender_3_batch_normalization_17_beta_v:	?L
8assignvariableop_63_adam_cnn__gender_3_dense_15_kernel_v:
?? D
6assignvariableop_64_adam_cnn__gender_3_dense_15_bias_v: J
8assignvariableop_65_adam_cnn__gender_3_dense_16_kernel_v: D
6assignvariableop_66_adam_cnn__gender_3_dense_16_bias_v:J
8assignvariableop_67_adam_cnn__gender_3_dense_17_kernel_v:D
6assignvariableop_68_adam_cnn__gender_3_dense_17_bias_v:
identity_70??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9? 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB+batchnorm1/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batchnorm1/beta/.ATTRIBUTES/VARIABLE_VALUEB1batchnorm1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batchnorm1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB+batchnorm2/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batchnorm2/beta/.ATTRIBUTES/VARIABLE_VALUEB1batchnorm2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batchnorm2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB+batchnorm3/gamma/.ATTRIBUTES/VARIABLE_VALUEB*batchnorm3/beta/.ATTRIBUTES/VARIABLE_VALUEB1batchnorm3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB5batchnorm3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCconv1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCconv3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAconv3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGbatchnorm3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFbatchnorm3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp/assignvariableop_cnn__gender_3_conv2d_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_cnn__gender_3_conv2d_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp=assignvariableop_2_cnn__gender_3_batch_normalization_15_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp<assignvariableop_3_cnn__gender_3_batch_normalization_15_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpCassignvariableop_4_cnn__gender_3_batch_normalization_15_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpGassignvariableop_5_cnn__gender_3_batch_normalization_15_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp1assignvariableop_6_cnn__gender_3_conv2d_16_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_cnn__gender_3_conv2d_16_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp=assignvariableop_8_cnn__gender_3_batch_normalization_16_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp<assignvariableop_9_cnn__gender_3_batch_normalization_16_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpDassignvariableop_10_cnn__gender_3_batch_normalization_16_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpHassignvariableop_11_cnn__gender_3_batch_normalization_16_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp2assignvariableop_12_cnn__gender_3_conv2d_17_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp0assignvariableop_13_cnn__gender_3_conv2d_17_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp>assignvariableop_14_cnn__gender_3_batch_normalization_17_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp=assignvariableop_15_cnn__gender_3_batch_normalization_17_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpDassignvariableop_16_cnn__gender_3_batch_normalization_17_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpHassignvariableop_17_cnn__gender_3_batch_normalization_17_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp1assignvariableop_18_cnn__gender_3_dense_15_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_cnn__gender_3_dense_15_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_cnn__gender_3_dense_16_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_cnn__gender_3_dense_16_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp1assignvariableop_22_cnn__gender_3_dense_17_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_cnn__gender_3_dense_17_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp9assignvariableop_33_adam_cnn__gender_3_conv2d_15_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_cnn__gender_3_conv2d_15_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpEassignvariableop_35_adam_cnn__gender_3_batch_normalization_15_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpDassignvariableop_36_adam_cnn__gender_3_batch_normalization_15_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp9assignvariableop_37_adam_cnn__gender_3_conv2d_16_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adam_cnn__gender_3_conv2d_16_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpEassignvariableop_39_adam_cnn__gender_3_batch_normalization_16_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpDassignvariableop_40_adam_cnn__gender_3_batch_normalization_16_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp9assignvariableop_41_adam_cnn__gender_3_conv2d_17_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_cnn__gender_3_conv2d_17_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpEassignvariableop_43_adam_cnn__gender_3_batch_normalization_17_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpDassignvariableop_44_adam_cnn__gender_3_batch_normalization_17_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp8assignvariableop_45_adam_cnn__gender_3_dense_15_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_cnn__gender_3_dense_15_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp8assignvariableop_47_adam_cnn__gender_3_dense_16_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adam_cnn__gender_3_dense_16_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp8assignvariableop_49_adam_cnn__gender_3_dense_17_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_cnn__gender_3_dense_17_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp9assignvariableop_51_adam_cnn__gender_3_conv2d_15_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_cnn__gender_3_conv2d_15_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpEassignvariableop_53_adam_cnn__gender_3_batch_normalization_15_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpDassignvariableop_54_adam_cnn__gender_3_batch_normalization_15_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp9assignvariableop_55_adam_cnn__gender_3_conv2d_16_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_cnn__gender_3_conv2d_16_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpEassignvariableop_57_adam_cnn__gender_3_batch_normalization_16_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpDassignvariableop_58_adam_cnn__gender_3_batch_normalization_16_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp9assignvariableop_59_adam_cnn__gender_3_conv2d_17_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adam_cnn__gender_3_conv2d_17_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpEassignvariableop_61_adam_cnn__gender_3_batch_normalization_17_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpDassignvariableop_62_adam_cnn__gender_3_batch_normalization_17_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp8assignvariableop_63_adam_cnn__gender_3_dense_15_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_cnn__gender_3_dense_15_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp8assignvariableop_65_adam_cnn__gender_3_dense_16_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_cnn__gender_3_dense_16_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_cnn__gender_3_dense_17_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_cnn__gender_3_dense_17_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_689
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_69?
Identity_70IdentityIdentity_69:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_70"#
identity_70Identity_70:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
C__inference_dense_17_layer_call_and_return_conditional_losses_17236

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_15_layer_call_fn_18767

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_175402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_19114

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17113

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????aa@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????aa@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_18805

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????aa@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????cc : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18886

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????aa@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????aa@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_16738

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18832

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_16606
input_1P
6cnn__gender_3_conv2d_15_conv2d_readvariableop_resource: E
7cnn__gender_3_conv2d_15_biasadd_readvariableop_resource: J
<cnn__gender_3_batch_normalization_15_readvariableop_resource: L
>cnn__gender_3_batch_normalization_15_readvariableop_1_resource: [
Mcnn__gender_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource: ]
Ocnn__gender_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: P
6cnn__gender_3_conv2d_16_conv2d_readvariableop_resource: @E
7cnn__gender_3_conv2d_16_biasadd_readvariableop_resource:@J
<cnn__gender_3_batch_normalization_16_readvariableop_resource:@L
>cnn__gender_3_batch_normalization_16_readvariableop_1_resource:@[
Mcnn__gender_3_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@]
Ocnn__gender_3_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@Q
6cnn__gender_3_conv2d_17_conv2d_readvariableop_resource:@?F
7cnn__gender_3_conv2d_17_biasadd_readvariableop_resource:	?K
<cnn__gender_3_batch_normalization_17_readvariableop_resource:	?M
>cnn__gender_3_batch_normalization_17_readvariableop_1_resource:	?\
Mcnn__gender_3_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?^
Ocnn__gender_3_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?I
5cnn__gender_3_dense_15_matmul_readvariableop_resource:
?? D
6cnn__gender_3_dense_15_biasadd_readvariableop_resource: G
5cnn__gender_3_dense_16_matmul_readvariableop_resource: D
6cnn__gender_3_dense_16_biasadd_readvariableop_resource:G
5cnn__gender_3_dense_17_matmul_readvariableop_resource:D
6cnn__gender_3_dense_17_biasadd_readvariableop_resource:
identity??Dcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?Fcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?3cnn__gender_3/batch_normalization_15/ReadVariableOp?5cnn__gender_3/batch_normalization_15/ReadVariableOp_1?Dcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?Fcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?3cnn__gender_3/batch_normalization_16/ReadVariableOp?5cnn__gender_3/batch_normalization_16/ReadVariableOp_1?Dcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?Fcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?3cnn__gender_3/batch_normalization_17/ReadVariableOp?5cnn__gender_3/batch_normalization_17/ReadVariableOp_1?.cnn__gender_3/conv2d_15/BiasAdd/ReadVariableOp?-cnn__gender_3/conv2d_15/Conv2D/ReadVariableOp?.cnn__gender_3/conv2d_16/BiasAdd/ReadVariableOp?-cnn__gender_3/conv2d_16/Conv2D/ReadVariableOp?.cnn__gender_3/conv2d_17/BiasAdd/ReadVariableOp?-cnn__gender_3/conv2d_17/Conv2D/ReadVariableOp?-cnn__gender_3/dense_15/BiasAdd/ReadVariableOp?,cnn__gender_3/dense_15/MatMul/ReadVariableOp?-cnn__gender_3/dense_16/BiasAdd/ReadVariableOp?,cnn__gender_3/dense_16/MatMul/ReadVariableOp?-cnn__gender_3/dense_17/BiasAdd/ReadVariableOp?,cnn__gender_3/dense_17/MatMul/ReadVariableOp?
-cnn__gender_3/conv2d_15/Conv2D/ReadVariableOpReadVariableOp6cnn__gender_3_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-cnn__gender_3/conv2d_15/Conv2D/ReadVariableOp?
cnn__gender_3/conv2d_15/Conv2DConv2Dinput_15cnn__gender_3/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2 
cnn__gender_3/conv2d_15/Conv2D?
.cnn__gender_3/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp7cnn__gender_3_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.cnn__gender_3/conv2d_15/BiasAdd/ReadVariableOp?
cnn__gender_3/conv2d_15/BiasAddBiasAdd'cnn__gender_3/conv2d_15/Conv2D:output:06cnn__gender_3/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2!
cnn__gender_3/conv2d_15/BiasAdd?
cnn__gender_3/conv2d_15/ReluRelu(cnn__gender_3/conv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
cnn__gender_3/conv2d_15/Relu?
3cnn__gender_3/batch_normalization_15/ReadVariableOpReadVariableOp<cnn__gender_3_batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype025
3cnn__gender_3/batch_normalization_15/ReadVariableOp?
5cnn__gender_3/batch_normalization_15/ReadVariableOp_1ReadVariableOp>cnn__gender_3_batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype027
5cnn__gender_3/batch_normalization_15/ReadVariableOp_1?
Dcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpMcnn__gender_3_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
Fcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOcnn__gender_3_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
5cnn__gender_3/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3*cnn__gender_3/conv2d_15/Relu:activations:0;cnn__gender_3/batch_normalization_15/ReadVariableOp:value:0=cnn__gender_3/batch_normalization_15/ReadVariableOp_1:value:0Lcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Ncnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 27
5cnn__gender_3/batch_normalization_15/FusedBatchNormV3?
&cnn__gender_3/max_pooling2d_15/MaxPoolMaxPool9cnn__gender_3/batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????cc *
ksize
*
paddingVALID*
strides
2(
&cnn__gender_3/max_pooling2d_15/MaxPool?
!cnn__gender_3/dropout_15/IdentityIdentity/cnn__gender_3/max_pooling2d_15/MaxPool:output:0*
T0*/
_output_shapes
:?????????cc 2#
!cnn__gender_3/dropout_15/Identity?
-cnn__gender_3/conv2d_16/Conv2D/ReadVariableOpReadVariableOp6cnn__gender_3_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-cnn__gender_3/conv2d_16/Conv2D/ReadVariableOp?
cnn__gender_3/conv2d_16/Conv2DConv2D*cnn__gender_3/dropout_15/Identity:output:05cnn__gender_3/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@*
paddingVALID*
strides
2 
cnn__gender_3/conv2d_16/Conv2D?
.cnn__gender_3/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp7cnn__gender_3_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.cnn__gender_3/conv2d_16/BiasAdd/ReadVariableOp?
cnn__gender_3/conv2d_16/BiasAddBiasAdd'cnn__gender_3/conv2d_16/Conv2D:output:06cnn__gender_3/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@2!
cnn__gender_3/conv2d_16/BiasAdd?
cnn__gender_3/conv2d_16/ReluRelu(cnn__gender_3/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa@2
cnn__gender_3/conv2d_16/Relu?
3cnn__gender_3/batch_normalization_16/ReadVariableOpReadVariableOp<cnn__gender_3_batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype025
3cnn__gender_3/batch_normalization_16/ReadVariableOp?
5cnn__gender_3/batch_normalization_16/ReadVariableOp_1ReadVariableOp>cnn__gender_3_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5cnn__gender_3/batch_normalization_16/ReadVariableOp_1?
Dcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpMcnn__gender_3_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
Fcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOcnn__gender_3_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
5cnn__gender_3/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3*cnn__gender_3/conv2d_16/Relu:activations:0;cnn__gender_3/batch_normalization_16/ReadVariableOp:value:0=cnn__gender_3/batch_normalization_16/ReadVariableOp_1:value:0Lcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Ncnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
is_training( 27
5cnn__gender_3/batch_normalization_16/FusedBatchNormV3?
&cnn__gender_3/max_pooling2d_16/MaxPoolMaxPool9cnn__gender_3/batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00@*
ksize
*
paddingVALID*
strides
2(
&cnn__gender_3/max_pooling2d_16/MaxPool?
!cnn__gender_3/dropout_16/IdentityIdentity/cnn__gender_3/max_pooling2d_16/MaxPool:output:0*
T0*/
_output_shapes
:?????????00@2#
!cnn__gender_3/dropout_16/Identity?
-cnn__gender_3/conv2d_17/Conv2D/ReadVariableOpReadVariableOp6cnn__gender_3_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02/
-cnn__gender_3/conv2d_17/Conv2D/ReadVariableOp?
cnn__gender_3/conv2d_17/Conv2DConv2D*cnn__gender_3/dropout_16/Identity:output:05cnn__gender_3/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?*
paddingVALID*
strides
2 
cnn__gender_3/conv2d_17/Conv2D?
.cnn__gender_3/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp7cnn__gender_3_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.cnn__gender_3/conv2d_17/BiasAdd/ReadVariableOp?
cnn__gender_3/conv2d_17/BiasAddBiasAdd'cnn__gender_3/conv2d_17/Conv2D:output:06cnn__gender_3/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?2!
cnn__gender_3/conv2d_17/BiasAdd?
cnn__gender_3/conv2d_17/ReluRelu(cnn__gender_3/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????..?2
cnn__gender_3/conv2d_17/Relu?
3cnn__gender_3/batch_normalization_17/ReadVariableOpReadVariableOp<cnn__gender_3_batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype025
3cnn__gender_3/batch_normalization_17/ReadVariableOp?
5cnn__gender_3/batch_normalization_17/ReadVariableOp_1ReadVariableOp>cnn__gender_3_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5cnn__gender_3/batch_normalization_17/ReadVariableOp_1?
Dcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpMcnn__gender_3_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
Fcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOcnn__gender_3_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
5cnn__gender_3/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3*cnn__gender_3/conv2d_17/Relu:activations:0;cnn__gender_3/batch_normalization_17/ReadVariableOp:value:0=cnn__gender_3/batch_normalization_17/ReadVariableOp_1:value:0Lcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Ncnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
is_training( 27
5cnn__gender_3/batch_normalization_17/FusedBatchNormV3?
&cnn__gender_3/max_pooling2d_17/MaxPoolMaxPool9cnn__gender_3/batch_normalization_17/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2(
&cnn__gender_3/max_pooling2d_17/MaxPool?
!cnn__gender_3/dropout_17/IdentityIdentity/cnn__gender_3/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:??????????2#
!cnn__gender_3/dropout_17/Identity?
cnn__gender_3/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
cnn__gender_3/flatten_5/Const?
cnn__gender_3/flatten_5/ReshapeReshape*cnn__gender_3/dropout_17/Identity:output:0&cnn__gender_3/flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2!
cnn__gender_3/flatten_5/Reshape?
,cnn__gender_3/dense_15/MatMul/ReadVariableOpReadVariableOp5cnn__gender_3_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02.
,cnn__gender_3/dense_15/MatMul/ReadVariableOp?
cnn__gender_3/dense_15/MatMulMatMul(cnn__gender_3/flatten_5/Reshape:output:04cnn__gender_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
cnn__gender_3/dense_15/MatMul?
-cnn__gender_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp6cnn__gender_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-cnn__gender_3/dense_15/BiasAdd/ReadVariableOp?
cnn__gender_3/dense_15/BiasAddBiasAdd'cnn__gender_3/dense_15/MatMul:product:05cnn__gender_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
cnn__gender_3/dense_15/BiasAdd?
cnn__gender_3/dense_15/ReluRelu'cnn__gender_3/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
cnn__gender_3/dense_15/Relu?
,cnn__gender_3/dense_16/MatMul/ReadVariableOpReadVariableOp5cnn__gender_3_dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,cnn__gender_3/dense_16/MatMul/ReadVariableOp?
cnn__gender_3/dense_16/MatMulMatMul)cnn__gender_3/dense_15/Relu:activations:04cnn__gender_3/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cnn__gender_3/dense_16/MatMul?
-cnn__gender_3/dense_16/BiasAdd/ReadVariableOpReadVariableOp6cnn__gender_3_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-cnn__gender_3/dense_16/BiasAdd/ReadVariableOp?
cnn__gender_3/dense_16/BiasAddBiasAdd'cnn__gender_3/dense_16/MatMul:product:05cnn__gender_3/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
cnn__gender_3/dense_16/BiasAdd?
cnn__gender_3/dense_16/ReluRelu'cnn__gender_3/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
cnn__gender_3/dense_16/Relu?
,cnn__gender_3/dense_17/MatMul/ReadVariableOpReadVariableOp5cnn__gender_3_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,cnn__gender_3/dense_17/MatMul/ReadVariableOp?
cnn__gender_3/dense_17/MatMulMatMul)cnn__gender_3/dense_16/Relu:activations:04cnn__gender_3/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cnn__gender_3/dense_17/MatMul?
-cnn__gender_3/dense_17/BiasAdd/ReadVariableOpReadVariableOp6cnn__gender_3_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-cnn__gender_3/dense_17/BiasAdd/ReadVariableOp?
cnn__gender_3/dense_17/BiasAddBiasAdd'cnn__gender_3/dense_17/MatMul:product:05cnn__gender_3/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
cnn__gender_3/dense_17/BiasAdd?
cnn__gender_3/dense_17/SigmoidSigmoid'cnn__gender_3/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
cnn__gender_3/dense_17/Sigmoid?
IdentityIdentity"cnn__gender_3/dense_17/Sigmoid:y:0E^cnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpG^cnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_14^cnn__gender_3/batch_normalization_15/ReadVariableOp6^cnn__gender_3/batch_normalization_15/ReadVariableOp_1E^cnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOpG^cnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_14^cnn__gender_3/batch_normalization_16/ReadVariableOp6^cnn__gender_3/batch_normalization_16/ReadVariableOp_1E^cnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOpG^cnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_14^cnn__gender_3/batch_normalization_17/ReadVariableOp6^cnn__gender_3/batch_normalization_17/ReadVariableOp_1/^cnn__gender_3/conv2d_15/BiasAdd/ReadVariableOp.^cnn__gender_3/conv2d_15/Conv2D/ReadVariableOp/^cnn__gender_3/conv2d_16/BiasAdd/ReadVariableOp.^cnn__gender_3/conv2d_16/Conv2D/ReadVariableOp/^cnn__gender_3/conv2d_17/BiasAdd/ReadVariableOp.^cnn__gender_3/conv2d_17/Conv2D/ReadVariableOp.^cnn__gender_3/dense_15/BiasAdd/ReadVariableOp-^cnn__gender_3/dense_15/MatMul/ReadVariableOp.^cnn__gender_3/dense_16/BiasAdd/ReadVariableOp-^cnn__gender_3/dense_16/MatMul/ReadVariableOp.^cnn__gender_3/dense_17/BiasAdd/ReadVariableOp-^cnn__gender_3/dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2?
Dcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOpDcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2?
Fcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1Fcnn__gender_3/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12j
3cnn__gender_3/batch_normalization_15/ReadVariableOp3cnn__gender_3/batch_normalization_15/ReadVariableOp2n
5cnn__gender_3/batch_normalization_15/ReadVariableOp_15cnn__gender_3/batch_normalization_15/ReadVariableOp_12?
Dcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOpDcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2?
Fcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Fcnn__gender_3/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12j
3cnn__gender_3/batch_normalization_16/ReadVariableOp3cnn__gender_3/batch_normalization_16/ReadVariableOp2n
5cnn__gender_3/batch_normalization_16/ReadVariableOp_15cnn__gender_3/batch_normalization_16/ReadVariableOp_12?
Dcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOpDcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2?
Fcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Fcnn__gender_3/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12j
3cnn__gender_3/batch_normalization_17/ReadVariableOp3cnn__gender_3/batch_normalization_17/ReadVariableOp2n
5cnn__gender_3/batch_normalization_17/ReadVariableOp_15cnn__gender_3/batch_normalization_17/ReadVariableOp_12`
.cnn__gender_3/conv2d_15/BiasAdd/ReadVariableOp.cnn__gender_3/conv2d_15/BiasAdd/ReadVariableOp2^
-cnn__gender_3/conv2d_15/Conv2D/ReadVariableOp-cnn__gender_3/conv2d_15/Conv2D/ReadVariableOp2`
.cnn__gender_3/conv2d_16/BiasAdd/ReadVariableOp.cnn__gender_3/conv2d_16/BiasAdd/ReadVariableOp2^
-cnn__gender_3/conv2d_16/Conv2D/ReadVariableOp-cnn__gender_3/conv2d_16/Conv2D/ReadVariableOp2`
.cnn__gender_3/conv2d_17/BiasAdd/ReadVariableOp.cnn__gender_3/conv2d_17/BiasAdd/ReadVariableOp2^
-cnn__gender_3/conv2d_17/Conv2D/ReadVariableOp-cnn__gender_3/conv2d_17/Conv2D/ReadVariableOp2^
-cnn__gender_3/dense_15/BiasAdd/ReadVariableOp-cnn__gender_3/dense_15/BiasAdd/ReadVariableOp2\
,cnn__gender_3/dense_15/MatMul/ReadVariableOp,cnn__gender_3/dense_15/MatMul/ReadVariableOp2^
-cnn__gender_3/dense_16/BiasAdd/ReadVariableOp-cnn__gender_3/dense_16/BiasAdd/ReadVariableOp2\
,cnn__gender_3/dense_16/MatMul/ReadVariableOp,cnn__gender_3/dense_16/MatMul/ReadVariableOp2^
-cnn__gender_3/dense_17/BiasAdd/ReadVariableOp-cnn__gender_3/dense_17/BiasAdd/ReadVariableOp2\
,cnn__gender_3/dense_17/MatMul/ReadVariableOp,cnn__gender_3/dense_17/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
)__inference_conv2d_17_layer_call_fn_18985

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????..?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_171422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????00@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_16_layer_call_fn_18938

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_174632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????aa@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????aa@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19039

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:?????????..?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????..?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????..?
 
_user_specified_nameinputs
??
?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18411
input_1B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: <
.batch_normalization_15_readvariableop_resource: >
0batch_normalization_15_readvariableop_1_resource: M
?batch_normalization_15_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@<
.batch_normalization_16_readvariableop_resource:@>
0batch_normalization_16_readvariableop_1_resource:@M
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_17_conv2d_readvariableop_resource:@?8
)conv2d_17_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_15_matmul_readvariableop_resource:
?? 6
(dense_15_biasadd_readvariableop_resource: 9
'dense_16_matmul_readvariableop_resource: 6
(dense_16_biasadd_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:6
(dense_17_biasadd_readvariableop_resource:
identity??%batch_normalization_15/AssignNewValue?'batch_normalization_15/AssignNewValue_1?6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_15/ReadVariableOp?'batch_normalization_15/ReadVariableOp_1?%batch_normalization_16/AssignNewValue?'batch_normalization_16/AssignNewValue_1?6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?%batch_normalization_17/AssignNewValue?'batch_normalization_17/AssignNewValue_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dinput_1'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/BiasAdd?
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_15/Relu?
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_15/ReadVariableOp?
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_15/ReadVariableOp_1?
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_15/FusedBatchNormV3?
%batch_normalization_15/AssignNewValueAssignVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource4batch_normalization_15/FusedBatchNormV3:batch_mean:07^batch_normalization_15/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_15/AssignNewValue?
'batch_normalization_15/AssignNewValue_1AssignVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_15/FusedBatchNormV3:batch_variance:09^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_15/AssignNewValue_1?
max_pooling2d_15/MaxPoolMaxPool+batch_normalization_15/FusedBatchNormV3:y:0*/
_output_shapes
:?????????cc *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPooly
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_15/dropout/Const?
dropout_15/dropout/MulMul!max_pooling2d_15/MaxPool:output:0!dropout_15/dropout/Const:output:0*
T0*/
_output_shapes
:?????????cc 2
dropout_15/dropout/Mul?
dropout_15/dropout/ShapeShape!max_pooling2d_15/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape?
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????cc *
dtype021
/dropout_15/dropout/random_uniform/RandomUniform?
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_15/dropout/GreaterEqual/y?
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????cc 2!
dropout_15/dropout/GreaterEqual?
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????cc 2
dropout_15/dropout/Cast?
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????cc 2
dropout_15/dropout/Mul_1?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2DConv2Ddropout_15/dropout/Mul_1:z:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@*
paddingVALID*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/BiasAdd~
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa@2
conv2d_16/Relu?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3conv2d_16/Relu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????aa@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_16/FusedBatchNormV3?
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_16/AssignNewValue?
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_16/AssignNewValue_1?
max_pooling2d_16/MaxPoolMaxPool+batch_normalization_16/FusedBatchNormV3:y:0*/
_output_shapes
:?????????00@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPooly
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_16/dropout/Const?
dropout_16/dropout/MulMul!max_pooling2d_16/MaxPool:output:0!dropout_16/dropout/Const:output:0*
T0*/
_output_shapes
:?????????00@2
dropout_16/dropout/Mul?
dropout_16/dropout/ShapeShape!max_pooling2d_16/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????00@*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform?
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_16/dropout/GreaterEqual/y?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????00@2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????00@2
dropout_16/dropout/Cast?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????00@2
dropout_16/dropout/Mul_1?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2DConv2Ddropout_16/dropout/Mul_1:z:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?*
paddingVALID*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/BiasAdd
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????..?2
conv2d_17/Relu?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%batch_normalization_17/ReadVariableOp?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype02)
'batch_normalization_17/ReadVariableOp_1?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3conv2d_17/Relu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????..?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_17/FusedBatchNormV3?
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_17/AssignNewValue?
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_17/AssignNewValue_1?
max_pooling2d_17/MaxPoolMaxPool+batch_normalization_17/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPooly
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMul!max_pooling2d_17/MaxPool:output:0!dropout_17/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_17/dropout/Mul?
dropout_17/dropout/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_17/dropout/Mul_1s
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
flatten_5/Const?
flatten_5/ReshapeReshapedropout_17/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_5/Reshape?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
?? *
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_15/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdd|
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Sigmoid?

IdentityIdentitydense_17/Sigmoid:y:0&^batch_normalization_15/AssignNewValue(^batch_normalization_15/AssignNewValue_17^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_15/AssignNewValue%batch_normalization_15/AssignNewValue2R
'batch_normalization_15/AssignNewValue_1'batch_normalization_15/AssignNewValue_12p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
6__inference_batch_normalization_16_layer_call_fn_18912

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_168102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_16766

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_17189

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_17129

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????00@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????00@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00@:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_16_layer_call_fn_18925

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_171132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????aa@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????aa@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????aa@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19021

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_16_layer_call_and_return_conditional_losses_17427

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????00@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????00@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????00@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????00@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????00@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????00@:W S
/
_output_shapes
:?????????00@
 
_user_specified_nameinputs
?
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_17181

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_cnn__gender_3_layer_call_fn_18623
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@?

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:
?? 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_176842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
c
E__inference_dropout_15_layer_call_and_return_conditional_losses_17077

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????cc 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????cc 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????cc :W S
/
_output_shapes
:?????????cc 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_17061

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:??????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_17_layer_call_fn_19083

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_169482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	conv1

batchnorm1
maxpool1
dropout1
	conv2

batchnorm2
maxpool2
dropout2
		conv3


batchnorm3
maxpool3
dropout3
flatten

dense1

dense2

dense3
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_model?{"name": "cnn__gender_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN_Gender", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 200, 200, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN_Gender"}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 0}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 200, 200, 3]}}
?

axis
	gamma
beta
 moving_mean
!moving_variance
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 198, 198, 32]}}
?
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 12}}
?
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 13}
?


.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 99, 99, 32]}}
?

4axis
	5gamma
6beta
7moving_mean
8moving_variance
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 21}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 97, 97, 64]}}
?
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 25}}
?
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 26}
?


Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 48, 48, 64]}}
?

Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 32}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 34}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 46, 46, 128]}}
?
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 38}}
?
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 39}
?
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 41}}
?

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 67712}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 67712]}}
?

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 46}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32]}}
?

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 50}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 51}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 16]}}
?
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem?m?m?m?.m?/m?5m?6m?Em?Fm?Lm?Mm?`m?am?fm?gm?lm?mm?v?v?v?v?.v?/v?5v?6v?Ev?Fv?Lv?Mv?`v?av?fv?gv?lv?mv?"
	optimizer
 "
trackable_list_wrapper
?
0
1
2
3
 4
!5
.6
/7
58
69
710
811
E12
F13
L14
M15
N16
O17
`18
a19
f20
g21
l22
m23"
trackable_list_wrapper
?
0
1
2
3
.4
/5
56
67
E8
F9
L10
M11
`12
a13
f14
g15
l16
m17"
trackable_list_wrapper
?
wlayer_regularization_losses
regularization_losses
	variables

xlayers
trainable_variables
ylayer_metrics
znon_trainable_variables
{metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
8:6 2cnn__gender_3/conv2d_15/kernel
*:( 2cnn__gender_3/conv2d_15/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
|layer_regularization_losses
regularization_losses
	variables

}layers
trainable_variables
~layer_metrics
non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8:6 2*cnn__gender_3/batch_normalization_15/gamma
7:5 2)cnn__gender_3/batch_normalization_15/beta
@:>  (20cnn__gender_3/batch_normalization_15/moving_mean
D:B  (24cnn__gender_3/batch_normalization_15/moving_variance
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 ?layer_regularization_losses
"regularization_losses
#	variables
?layers
$trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
&regularization_losses
'	variables
?layers
(trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
*regularization_losses
+	variables
?layers
,trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
8:6 @2cnn__gender_3/conv2d_16/kernel
*:(@2cnn__gender_3/conv2d_16/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
 ?layer_regularization_losses
0regularization_losses
1	variables
?layers
2trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8:6@2*cnn__gender_3/batch_normalization_16/gamma
7:5@2)cnn__gender_3/batch_normalization_16/beta
@:>@ (20cnn__gender_3/batch_normalization_16/moving_mean
D:B@ (24cnn__gender_3/batch_normalization_16/moving_variance
 "
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
 ?layer_regularization_losses
9regularization_losses
:	variables
?layers
;trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
=regularization_losses
>	variables
?layers
?trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Aregularization_losses
B	variables
?layers
Ctrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:7@?2cnn__gender_3/conv2d_17/kernel
+:)?2cnn__gender_3/conv2d_17/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Gregularization_losses
H	variables
?layers
Itrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
9:7?2*cnn__gender_3/batch_normalization_17/gamma
8:6?2)cnn__gender_3/batch_normalization_17/beta
A:?? (20cnn__gender_3/batch_normalization_17/moving_mean
E:C? (24cnn__gender_3/batch_normalization_17/moving_variance
 "
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Pregularization_losses
Q	variables
?layers
Rtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Tregularization_losses
U	variables
?layers
Vtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Xregularization_losses
Y	variables
?layers
Ztrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
\regularization_losses
]	variables
?layers
^trainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
1:/
?? 2cnn__gender_3/dense_15/kernel
):' 2cnn__gender_3/dense_15/bias
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
?
 ?layer_regularization_losses
bregularization_losses
c	variables
?layers
dtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:- 2cnn__gender_3/dense_16/kernel
):'2cnn__gender_3/dense_16/bias
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
 ?layer_regularization_losses
hregularization_losses
i	variables
?layers
jtrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-2cnn__gender_3/dense_17/kernel
):'2cnn__gender_3/dense_17/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
 ?layer_regularization_losses
nregularization_losses
o	variables
?layers
ptrainable_variables
?layer_metrics
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_dict_wrapper
J
 0
!1
72
83
N4
O5"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 54}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 0}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
=:; 2%Adam/cnn__gender_3/conv2d_15/kernel/m
/:- 2#Adam/cnn__gender_3/conv2d_15/bias/m
=:; 21Adam/cnn__gender_3/batch_normalization_15/gamma/m
<:: 20Adam/cnn__gender_3/batch_normalization_15/beta/m
=:; @2%Adam/cnn__gender_3/conv2d_16/kernel/m
/:-@2#Adam/cnn__gender_3/conv2d_16/bias/m
=:;@21Adam/cnn__gender_3/batch_normalization_16/gamma/m
<::@20Adam/cnn__gender_3/batch_normalization_16/beta/m
>:<@?2%Adam/cnn__gender_3/conv2d_17/kernel/m
0:.?2#Adam/cnn__gender_3/conv2d_17/bias/m
>:<?21Adam/cnn__gender_3/batch_normalization_17/gamma/m
=:;?20Adam/cnn__gender_3/batch_normalization_17/beta/m
6:4
?? 2$Adam/cnn__gender_3/dense_15/kernel/m
.:, 2"Adam/cnn__gender_3/dense_15/bias/m
4:2 2$Adam/cnn__gender_3/dense_16/kernel/m
.:,2"Adam/cnn__gender_3/dense_16/bias/m
4:22$Adam/cnn__gender_3/dense_17/kernel/m
.:,2"Adam/cnn__gender_3/dense_17/bias/m
=:; 2%Adam/cnn__gender_3/conv2d_15/kernel/v
/:- 2#Adam/cnn__gender_3/conv2d_15/bias/v
=:; 21Adam/cnn__gender_3/batch_normalization_15/gamma/v
<:: 20Adam/cnn__gender_3/batch_normalization_15/beta/v
=:; @2%Adam/cnn__gender_3/conv2d_16/kernel/v
/:-@2#Adam/cnn__gender_3/conv2d_16/bias/v
=:;@21Adam/cnn__gender_3/batch_normalization_16/gamma/v
<::@20Adam/cnn__gender_3/batch_normalization_16/beta/v
>:<@?2%Adam/cnn__gender_3/conv2d_17/kernel/v
0:.?2#Adam/cnn__gender_3/conv2d_17/bias/v
>:<?21Adam/cnn__gender_3/batch_normalization_17/gamma/v
=:;?20Adam/cnn__gender_3/batch_normalization_17/beta/v
6:4
?? 2$Adam/cnn__gender_3/dense_15/kernel/v
.:, 2"Adam/cnn__gender_3/dense_15/bias/v
4:2 2$Adam/cnn__gender_3/dense_16/kernel/v
.:,2"Adam/cnn__gender_3/dense_16/bias/v
4:22$Adam/cnn__gender_3/dense_17/kernel/v
.:,2"Adam/cnn__gender_3/dense_17/bias/v
?2?
 __inference__wrapped_model_16606?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_1???????????
?2?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18081
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18198
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18294
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18411?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_cnn__gender_3_layer_call_fn_18464
-__inference_cnn__gender_3_layer_call_fn_18517
-__inference_cnn__gender_3_layer_call_fn_18570
-__inference_cnn__gender_3_layer_call_fn_18623?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_15_layer_call_and_return_conditional_losses_18634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_15_layer_call_fn_18643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18661
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18679
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18697
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18715?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_15_layer_call_fn_18728
6__inference_batch_normalization_15_layer_call_fn_18741
6__inference_batch_normalization_15_layer_call_fn_18754
6__inference_batch_normalization_15_layer_call_fn_18767?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_16738?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_15_layer_call_fn_16744?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_dropout_15_layer_call_and_return_conditional_losses_18772
E__inference_dropout_15_layer_call_and_return_conditional_losses_18784?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_15_layer_call_fn_18789
*__inference_dropout_15_layer_call_fn_18794?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_18805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_16_layer_call_fn_18814?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18832
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18850
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18868
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18886?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_16_layer_call_fn_18899
6__inference_batch_normalization_16_layer_call_fn_18912
6__inference_batch_normalization_16_layer_call_fn_18925
6__inference_batch_normalization_16_layer_call_fn_18938?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_16876?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_16_layer_call_fn_16882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_dropout_16_layer_call_and_return_conditional_losses_18943
E__inference_dropout_16_layer_call_and_return_conditional_losses_18955?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_16_layer_call_fn_18960
*__inference_dropout_16_layer_call_fn_18965?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_18976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_17_layer_call_fn_18985?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19003
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19021
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19039
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19057?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_batch_normalization_17_layer_call_fn_19070
6__inference_batch_normalization_17_layer_call_fn_19083
6__inference_batch_normalization_17_layer_call_fn_19096
6__inference_batch_normalization_17_layer_call_fn_19109?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_17014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_17_layer_call_fn_17020?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_dropout_17_layer_call_and_return_conditional_losses_19114
E__inference_dropout_17_layer_call_and_return_conditional_losses_19126?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_17_layer_call_fn_19131
*__inference_dropout_17_layer_call_fn_19136?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_flatten_5_layer_call_and_return_conditional_losses_19142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_5_layer_call_fn_19147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_15_layer_call_and_return_conditional_losses_19158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_15_layer_call_fn_19167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_16_layer_call_and_return_conditional_losses_19178?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_16_layer_call_fn_19187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_17_layer_call_and_return_conditional_losses_19198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_17_layer_call_fn_19207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_17985input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_16606? !./5678EFLMNO`afglm:?7
0?-
+?(
input_1???????????
? "3?0
.
output_1"?
output_1??????????
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18661? !M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18679? !M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18697v !=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
Q__inference_batch_normalization_15_layer_call_and_return_conditional_losses_18715v !=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
6__inference_batch_normalization_15_layer_call_fn_18728? !M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_15_layer_call_fn_18741? !M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_15_layer_call_fn_18754i !=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
6__inference_batch_normalization_15_layer_call_fn_18767i !=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18832?5678M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18850?5678M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18868r5678;?8
1?.
(?%
inputs?????????aa@
p 
? "-?*
#? 
0?????????aa@
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18886r5678;?8
1?.
(?%
inputs?????????aa@
p
? "-?*
#? 
0?????????aa@
? ?
6__inference_batch_normalization_16_layer_call_fn_18899?5678M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_16_layer_call_fn_18912?5678M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_16_layer_call_fn_18925e5678;?8
1?.
(?%
inputs?????????aa@
p 
? " ??????????aa@?
6__inference_batch_normalization_16_layer_call_fn_18938e5678;?8
1?.
(?%
inputs?????????aa@
p
? " ??????????aa@?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19003?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19021?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19039tLMNO<?9
2?/
)?&
inputs?????????..?
p 
? ".?+
$?!
0?????????..?
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_19057tLMNO<?9
2?/
)?&
inputs?????????..?
p
? ".?+
$?!
0?????????..?
? ?
6__inference_batch_normalization_17_layer_call_fn_19070?LMNON?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_17_layer_call_fn_19083?LMNON?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_17_layer_call_fn_19096gLMNO<?9
2?/
)?&
inputs?????????..?
p 
? "!??????????..??
6__inference_batch_normalization_17_layer_call_fn_19109gLMNO<?9
2?/
)?&
inputs?????????..?
p
? "!??????????..??
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18081? !./5678EFLMNO`afglm=?:
3?0
*?'
inputs???????????
p 
? "%?"
?
0?????????
? ?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18198? !./5678EFLMNO`afglm=?:
3?0
*?'
inputs???????????
p
? "%?"
?
0?????????
? ?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18294? !./5678EFLMNO`afglm>?;
4?1
+?(
input_1???????????
p 
? "%?"
?
0?????????
? ?
H__inference_cnn__gender_3_layer_call_and_return_conditional_losses_18411? !./5678EFLMNO`afglm>?;
4?1
+?(
input_1???????????
p
? "%?"
?
0?????????
? ?
-__inference_cnn__gender_3_layer_call_fn_18464t !./5678EFLMNO`afglm>?;
4?1
+?(
input_1???????????
p 
? "???????????
-__inference_cnn__gender_3_layer_call_fn_18517s !./5678EFLMNO`afglm=?:
3?0
*?'
inputs???????????
p 
? "???????????
-__inference_cnn__gender_3_layer_call_fn_18570s !./5678EFLMNO`afglm=?:
3?0
*?'
inputs???????????
p
? "???????????
-__inference_cnn__gender_3_layer_call_fn_18623t !./5678EFLMNO`afglm>?;
4?1
+?(
input_1???????????
p
? "???????????
D__inference_conv2d_15_layer_call_and_return_conditional_losses_18634p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_15_layer_call_fn_18643c9?6
/?,
*?'
inputs???????????
? ""???????????? ?
D__inference_conv2d_16_layer_call_and_return_conditional_losses_18805l./7?4
-?*
(?%
inputs?????????cc 
? "-?*
#? 
0?????????aa@
? ?
)__inference_conv2d_16_layer_call_fn_18814_./7?4
-?*
(?%
inputs?????????cc 
? " ??????????aa@?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_18976mEF7?4
-?*
(?%
inputs?????????00@
? ".?+
$?!
0?????????..?
? ?
)__inference_conv2d_17_layer_call_fn_18985`EF7?4
-?*
(?%
inputs?????????00@
? "!??????????..??
C__inference_dense_15_layer_call_and_return_conditional_losses_19158^`a1?.
'?$
"?
inputs???????????
? "%?"
?
0????????? 
? }
(__inference_dense_15_layer_call_fn_19167Q`a1?.
'?$
"?
inputs???????????
? "?????????? ?
C__inference_dense_16_layer_call_and_return_conditional_losses_19178\fg/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_16_layer_call_fn_19187Ofg/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_17_layer_call_and_return_conditional_losses_19198\lm/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_17_layer_call_fn_19207Olm/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dropout_15_layer_call_and_return_conditional_losses_18772l;?8
1?.
(?%
inputs?????????cc 
p 
? "-?*
#? 
0?????????cc 
? ?
E__inference_dropout_15_layer_call_and_return_conditional_losses_18784l;?8
1?.
(?%
inputs?????????cc 
p
? "-?*
#? 
0?????????cc 
? ?
*__inference_dropout_15_layer_call_fn_18789_;?8
1?.
(?%
inputs?????????cc 
p 
? " ??????????cc ?
*__inference_dropout_15_layer_call_fn_18794_;?8
1?.
(?%
inputs?????????cc 
p
? " ??????????cc ?
E__inference_dropout_16_layer_call_and_return_conditional_losses_18943l;?8
1?.
(?%
inputs?????????00@
p 
? "-?*
#? 
0?????????00@
? ?
E__inference_dropout_16_layer_call_and_return_conditional_losses_18955l;?8
1?.
(?%
inputs?????????00@
p
? "-?*
#? 
0?????????00@
? ?
*__inference_dropout_16_layer_call_fn_18960_;?8
1?.
(?%
inputs?????????00@
p 
? " ??????????00@?
*__inference_dropout_16_layer_call_fn_18965_;?8
1?.
(?%
inputs?????????00@
p
? " ??????????00@?
E__inference_dropout_17_layer_call_and_return_conditional_losses_19114n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
E__inference_dropout_17_layer_call_and_return_conditional_losses_19126n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
*__inference_dropout_17_layer_call_fn_19131a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
*__inference_dropout_17_layer_call_fn_19136a<?9
2?/
)?&
inputs??????????
p
? "!????????????
D__inference_flatten_5_layer_call_and_return_conditional_losses_19142c8?5
.?+
)?&
inputs??????????
? "'?$
?
0???????????
? ?
)__inference_flatten_5_layer_call_fn_19147V8?5
.?+
)?&
inputs??????????
? "?????????????
K__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_16738?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_15_layer_call_fn_16744?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_16876?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_16_layer_call_fn_16882?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_17014?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_17_layer_call_fn_17020?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
#__inference_signature_wrapper_17985? !./5678EFLMNO`afglmE?B
? 
;?8
6
input_1+?(
input_1???????????"3?0
.
output_1"?
output_1?????????