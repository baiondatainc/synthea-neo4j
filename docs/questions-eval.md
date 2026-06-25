## 1. Patient Lookup & Profile

1. Show me the patient with ID 1000000 in RADM. **[E]**
2. What's Mary Allen's date of birth? **[E]**
3. Find the patient whose phone number is 512-399-2181. **[E]**
4. List all patients in Houston, TX. **[E]**
5. Pull up the patient with email sandrabro452@yahoo.com. **[E]**
6. Who is the responsible party for patient 1000000? **[E]**
7. How many female patients over the age of 70 do we have? **[M]**
8. Show me all patients with a bad address flag. **[E]**
9. How many patients live in Texas vs Florida? **[M]**
10. Find patients whose responsible party has a different last name than the patient. **[M]**

## 2. Visit History & Encounters

11. How many visits has patient 1000000 had? **[E]**
12. Show me Mary Allen's last visit. **[E]**
13. What was the most recent visit at location ESR01? **[M]**
14. List all visits in March 2026. **[E]**
15. Which patients have had more than 10 visits? **[M]**
16. How many visits were there at the Houston location last year? **[M]**
17. Average number of visits per patient. **[M]**
18. Show me visits where the admit date and discharge date are the same. **[E]**
19. Find patients whose first visit was over 5 years ago but had a visit this year. **[H]**
20. Which visits had three different insurance plans on them? **[M]**

## 3. Billing & Charges

21. What's the total charged for patient 1000000? **[E]**
22. Show me charges over $1,000. **[E]**
23. List voided charges from this year. **[E]**
24. How many charges are currently on hold? **[E]**
25. Total amount billed across all practices. **[M]**
26. Total billed for RADM specifically. **[E]**
27. Top 10 most expensive charges. **[E]**
28. Show me charges with a balance that exceeds the original charge amount. **[M]**
29. How many charges came through with the NM modality? **[E]**
30. Average charge amount by modality. **[M]**

## 4. Payments & Transactions

31. How much has patient 1000000 paid in total? **[E]**
32. List all payments made in April 2026. **[E]**
33. Show me refunds issued this year. **[M]**
34. How many transactions were ERS payments? **[E]**
35. Total collected by IVR vs manual payments. **[M]**
36. What's the breakdown of transactions by payment method? **[E]**
37. Which charges have multiple payment transactions against them? **[M]**
38. Show me transactions where the payment came from the secondary payer. **[H]**
39. Average days between a charge being posted and the first payment hitting it. **[H]**
40. List the largest payments received this quarter. **[E]**

## 5. Outstanding Balances & A/R

41. What's the total outstanding across all patients? **[E]**
42. Show me patients with a balance over $5,000. **[E]**
43. How much is sitting in 361+ day aging? **[E]**
44. Break down A/R by aging bucket. **[M]**
45. Top 20 patients by outstanding balance. **[E]**
46. What percentage of our A/R is over 180 days old? **[M]**
47. How much patient-pay balance do we have right now? **[M]**
48. Show me the A/R aging for RADX only. **[M]**
49. Which location has the worst aging profile? **[H]**
50. How much of the outstanding balance belongs to patients flagged as catastrophe? **[H]**

## 6. Collections Operations

51. List patients with bad debt adjustments this year. **[M]**
52. How many accounts have been placed with a collection agency? **[E]**
53. Show me patients in the catastrophe cohort. **[E]**
54. Average days to agency placement. **[M]**
55. Which patients have received three or more statements but still haven't paid? **[H]**
56. Show me the high-balance self-pay patients we haven't called yet. **[H]**
57. Patients with payment plans currently in place. **[E]**
58. Who's eligible for charity care that hasn't received it? **[H]**
59. How many bad debt write-offs happened last month vs this month? **[M]**
60. Compare recovery rate between the clean cohort and the friction cohort. **[H]**

## 7. Statements & Communications

61. How many statements went out this month? **[E]**
62. Show me Mary Allen's statement history. **[E]**
63. How many statements failed all three delivery methods? **[M]**
64. List patients who only get statements by mail (no email or text). **[M]**
65. What percentage of our statements get delivered by email? **[M]**
66. Average number of statements sent before payment. **[H]**
67. How many statements are sitting on hold right now? **[E]**
68. Patients who got a Statement 3 or higher level. **[M]**
69. Which patients received a statement but never opened any prior communication? **[H]**
70. Time between consecutive statements per patient. **[M]**

## 8. Insurance & Payer Analysis

71. Who are our top 10 carriers by volume? **[M]**
72. What's our payor mix? **[M]**
73. Show me all visits where Cigna was the primary payer. **[E]**
74. How many patients have Medicare? **[M]**
75. List Commercial HMO plans we work with. **[E]**
76. Which carrier pays the fastest? **[H]**
77. Average reimbursement rate by carrier. **[H]**
78. How many visits have a secondary insurance? **[E]**
79. Patients whose primary insurance changed between visits. **[H]**
80. Carrier with the highest denial rate. **[H]**

## 9. Denials & Adjustments

81. How many transactions had a denial code last month? **[E]**
82. Most common denial reason. **[M]**
83. Total contractual write-offs this year. **[E]**
84. Breakdown of adjustments by bucket. **[M]**
85. Show me CO45 denials. **[E]**
86. Which providers have the highest denial rate on their referrals? **[H]**
87. Procedures most often denied. **[H]**
88. Patients with refund reversals. **[M]**
89. How much have we written off as charity care? **[E]**
90. Compare denial patterns between Medicare and commercial. **[H]**

## 10. Provider & Referral Patterns

91. Top 10 referring physicians by volume. **[M]**
92. Who's our highest-billing rendering radiologist? **[M]**
93. Show me the providers who both refer and render. **[H]**
94. Which referring doctor's charges most often end up in bad debt? **[H]**
95. List providers we've seen this year but not last year. **[H]**
96. How many distinct providers do we have? **[E]**
97. Provider with the most ICD-10 diversity on their charges. **[H]**
98. Referring physicians whose patients have the highest self-pay rate. **[H]**
99. Are there any provider IDs that show up as "0"? **[E]**
100. Compare rendering doctors by average charge amount. **[M]**

## 11. Procedures & Clinical Mix

101. What's our top 10 procedures by volume? **[M]**
102. How many MRI charges did we have this year? **[M]**
103. Modality mix across all practices. **[M]**
104. Most expensive procedure on average. **[M]**
105. Procedures we only do at one location. **[H]**
106. Volume of CT scans by month. **[M]**
107. Which procedures have the longest time to payment? **[H]**
108. How many distinct CPT codes have we billed? **[E]**
109. Show me procedures with denial rates above 20%. **[H]**
110. Average charge per modality. **[M]**

## 12. Diagnoses & Case Mix

111. Top 20 ICD-10 codes by frequency. **[M]**
112. How many charges have a primary cardiovascular diagnosis? **[H]**
113. Distribution of charges by ICD-10 chapter. **[M]**
114. Patients with chronic back pain diagnoses. **[M]**
115. Are there any charges using invalid ICD-10 codes? **[M]**
116. Most common diagnosis paired with chest CT. **[H]**
117. How many diagnoses on average per charge? **[M]**
118. Charges with 5+ diagnoses on them. **[E]**
119. Patients with diagnoses from both circulatory and respiratory chapters in the same visit. **[H]**
120. Diagnosis chapters most associated with the friction cohort. **[H]**

## 13. Location & Practice Performance

121. Total billed by location. **[M]**
122. Which location sees the most patients? **[M]**
123. Compare RADM and RADX on total revenue. **[M]**
124. Locations in Florida vs Texas — which performs better? **[H]**
125. How many locations don't have an NPI on file? **[E]**
126. Average patient balance per location. **[M]**
127. Which location has the youngest patient population? **[M]**
128. Show me locations where service and interpretation are at different sites. **[H]**
129. Locations with no reviews in Birdeye. **[M]**
130. Top location by self-pay percentage. **[H]**

## 14. Contact Center Operations

131. How many calls came in last week? **[E]**
132. Top 10 agents by call volume. **[M]**
133. Average handle time per agent. **[M]**
134. Calls that went to voicemail. **[E]**
135. Abandonment rate across all queues. **[M]**
136. Which team has the worst hold times? **[M]**
137. Show me agents handling collections calls. **[M]**
138. How many inbound vs outbound calls did we do? **[E]**
139. Skill routing — what does PMR English get most? **[M]**
140. Repeat callers in the last 30 days. **[H]**

## 15. Campaign & Outreach

141. Which campaign is generating the most calls? **[M]**
142. SAPA campaign — how many patients have we reached? **[M]**
143. Outreach effectiveness — calls per patient by campaign. **[H]**
144. Patients with no campaign assignment yet. **[E]**
145. Compare contact rate across campaigns. **[H]**
146. How many patients are in the MRB campaign? **[M]**
147. Best-performing campaign by recovery dollars. **[H]**
148. Campaigns running in the last 90 days. **[M]**
149. Patients in multiple campaigns simultaneously. **[H]**
150. Average calls before a patient pays, by campaign. **[H]**

## 16. Cross-Practice & Patient Identity

151. How many patients are seen at more than one practice? **[M]**
152. Show me Mary Allen's full activity across all practices. **[H]**
153. Patients who appear in RADM and RADX with the same name. **[H]**
154. Multi-practice patients with high total balance. **[H]**
155. Which combinations of practices share the most patients? **[H]**
156. Patient 1000000 — is this person also a patient anywhere else? **[M]**
157. How many distinct humans do we serve in total? **[M]**
158. Cross-practice patients with calls at one practice but billing at another. **[H]**
159. Are there patients flagged as multi-practice but only show up in one source? **[H]**
160. Patients whose demographics don't match across their practice identities. **[H]**

## 17. Cohort & Segment Analysis

161. Break down patients by cohort. **[M]**
162. How big is the catastrophe cohort? **[E]**
163. Clean cohort recovery rate. **[H]**
164. SAPA patients — current outstanding. **[M]**
165. Friction cohort — average days in A/R. **[H]**
166. NRAA patients vs Tennessee patients — which collects better? **[H]**
167. Patients in the Atlanta 404 cohort. **[E]**
168. Self-pay patients in the catastrophe cohort. **[M]**
169. Are any patients flagged in both clean and friction? **[M]**
170. Cohort distribution by location. **[H]**

## 18. Trend & Temporal Analysis

171. Month-over-month charge volume this year. **[M]**
172. Year-over-year comparison of total collections. **[M]**
173. Show me revenue trend for the last 12 months. **[M]**
174. Are denials going up or down quarter over quarter? **[H]**
175. Average days in A/R trended over time. **[H]**
176. Statement volume by month. **[M]**
177. Call volume seasonality — which month is busiest? **[M]**
178. Patients acquired this year vs last year. **[M]**
179. Time from first visit to first payment, trended by quarter. **[H]**
180. Bad debt write-off trend over the past 18 months. **[H]**

## 19. Executive / KPI Dashboard

181. Give me a summary of today. **[H]**
182. What's our cash collection rate this year? **[H]**
183. Net collection rate per location. **[H]**
184. Top 5 KPIs for last month. **[H]**
185. How are we doing vs last quarter? **[H]**
186. Practice-level scorecard for RADM. **[H]**
187. Total revenue, total adjustments, total outstanding — single view. **[M]**
188. Which practice is our most profitable? **[H]**
189. Show me the executive dashboard. **[H]**
190. Where are we leaking money? **[H]**

## 20. Complex / Multi-Hop & Edge Cases

191. Patients who called us before we even sent them a statement. **[H]**
192. Find people whose referring doctor is also their rendering doctor. **[H]**
193. Which patients had a low Birdeye rating at their primary location and then went to bad debt? **[H]**
194. Show me cases where a patient was contacted by an agent on a campaign their phone number isn't even in. **[H]**
195. Find visits that had three payers, all in the same carrier family. **[H]**
196. Patients with a Spanish-skill call who later paid via IVR. **[H]**
197. Charges where the diagnosis chapter changed between primary and secondary positions. **[H]**
198. Identify potential PHI leaks in our Birdeye reviews. **[M]**
199. Patients whose responsible party DOB is older than the patient's by less than 10 years (likely a partner, not a parent). **[H]**
200. The patient — show me everything about her. **[H]** *(deliberately ambiguous — tests how the model handles missing referent)*