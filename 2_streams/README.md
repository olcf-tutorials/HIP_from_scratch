# Streams and Events

## Streams
Thus far, we have seen only the execution of one kernel launch or one
`Host<->Device` memory copy at a time. If the kernel is not saturating the GPU with the
work it is doing, and if your code has work that could be overlapped, you could take
advantage of **HIP Streams**. 

The idea here is that you can create different pipes or _streams_ where to which you can
assign different tasks that don't depend on each other and thus could execute at the same
time and make better use of the GPU. Let us look at a couple of code examples to make this
concrete (TODO: make code examples, make an example where you're overlapping multiple
hipblasdgemm calls with stream)

(TODO: make example where streams aren't useful because the whole GPU is being saturated
and using stream doesn't really make it go faster).


## Events
Allows you to synchronize a stream with another stream. So we can wait for an event happen
in one stream to trigger something else starting in another stream.


TODOs:
What is pinned memory?
Host data allocations are pageable by default. HIP can instruct data to be pinned instead
to allow direct access of memory from GPU. (TODO: need more info on why this is
useful). `Host<->Device` memcpy bandwidth will increase if using pinned memory. So it's
good practice to use pinned memory when there is frequent data transfers.

(TODO: show an example where pinned memory allows for a faster run)


## Resources:
GPU Programming Concepts (Part 2) (covers streams): https://www.youtube.com/watch?v=i0GzebZKi10
