import { addDays, localISODate, startOfDay } from "../lib/progressStore";

const WEEKS = 18;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function level(count: number): number {
  if (count <= 0) return 0;
  if (count === 1) return 1;
  if (count === 2) return 2;
  if (count < 5) return 3;
  return 4;
}

export function Heatmap({ activity }: { activity: Record<string, number> }) {
  const today = startOfDay(new Date());
  const mondayOffset = (today.getDay() + 6) % 7;
  const thisMonday = addDays(today, -mondayOffset);
  const start = addDays(thisMonday, -7 * (WEEKS - 1));
  const columns = Array.from({ length: WEEKS }, (_, week) =>
    Array.from({ length: 7 }, (_, day) => {
      const date = addDays(start, week * 7 + day);
      const key = localISODate(date);
      return { date, key, count: activity[key] ?? 0, future: date.getTime() > today.getTime() };
    }),
  );

  return (
    <div className="heatmap" aria-label="Activity over the last 18 weeks">
      <div className="heatmap-months" aria-hidden="true">
        {columns.map((week, index) => {
          const first = week[0].date;
          const show = index === 0 || first.getDate() <= 7;
          return <span key={week[0].key}>{show ? MONTHS[first.getMonth()] : ""}</span>;
        })}
      </div>
      <div className="heatmap-grid">
        <div className="heatmap-days" aria-hidden="true">
          <span>Mon</span>
          <span>Wed</span>
          <span>Fri</span>
        </div>
        <div className="heatmap-weeks">
          {columns.map((week) => (
            <div key={week[0].key} className="heatmap-week">
              {week.map((cell) => (
                <span
                  key={cell.key}
                  className={`heat heat-${cell.future ? 0 : level(cell.count)}`}
                  title={`${cell.key}: ${cell.count} study action${cell.count === 1 ? "" : "s"}`}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="heatmap-legend">
        <span>Less</span>
        {[0, 1, 2, 3, 4].map((value) => (
          <span key={value} className={`heat heat-${value}`} />
        ))}
        <span>More</span>
      </div>
    </div>
  );
}
