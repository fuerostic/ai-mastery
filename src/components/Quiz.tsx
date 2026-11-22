import type { QuizQuestion } from "../content/types";
import { useProgress } from "../lib/progress";
import { quizCorrectCount } from "../lib/progressStore";
import { InlineMarkdown } from "./MarkdownView";

export function Quiz({ lessonId, questions }: { lessonId: string; questions: QuizQuestion[] }) {
  const { data, answerQuiz } = useProgress();
  const answers = data.quiz[lessonId] ?? {};
  const correct = quizCorrectCount({ id: lessonId, quiz: questions }, data);

  return (
    <div className="quiz">
      <p className="quiz-score">
        {correct} of {questions.length} correct. You can change an answer.
      </p>
      <ol>
        {questions.map((question, index) => {
          const selected = answers[question.id];
          const answered = selected !== undefined;
          return (
            <li key={question.id} className="quiz-card">
              <div className="quiz-prompt">
                <span className="quiz-kicker">Question {index + 1}</span>
                <InlineMarkdown source={question.prompt} />
              </div>
              <div className="choices" role="radiogroup" aria-label={`Question ${index + 1}`}>
                {question.choices.map((choice, choiceIndex) => {
                  const isSelected = selected === choiceIndex;
                  const isAnswer = choiceIndex === question.answer;
                  let state = "";
                  if (answered && isAnswer) state = "correct";
                  else if (answered && isSelected) state = "wrong";
                  return (
                    <label key={choice} className={`choice ${state}`}>
                      <input
                        type="radio"
                        name={`${lessonId}-${question.id}`}
                        checked={isSelected}
                        onChange={() => answerQuiz(lessonId, question.id, choiceIndex)}
                      />
                      <InlineMarkdown source={choice} />
                    </label>
                  );
                })}
              </div>
              {answered ? (
                <p className={selected === question.answer ? "explain good" : "explain bad"}>
                  <InlineMarkdown source={question.explanation} />
                </p>
              ) : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
