def longestSubarray(nums):
    max_number = max(nums)
    i = 0
    ans = 1
    while i < len(nums):

        if nums[i] == max_number:
            start = i

            while i < len(nums) and nums[i] == max_number:
                i += 1


            ans = max(ans, i - start)

        i += 1